"""Upload storage: local backend, registry, and S3 backend via a fake client."""
import io
import sys
import tempfile
import types
import unittest

from core import storage
from core.storage import (
    LocalUploadStore,
    S3UploadStore,
    build_upload_store,
    get_upload_store,
    reset_upload_store,
)


class TestLocalStorage(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.mkdtemp(prefix="mcq_storage_")
        import config

        self._saved = config.UPLOAD_FOLDER
        config.UPLOAD_FOLDER = self._folder
        reset_upload_store()

    def tearDown(self):
        import config

        config.UPLOAD_FOLDER = self._saved
        reset_upload_store()

    def test_save_open_exists_delete(self):
        store = LocalUploadStore(self._folder)
        data = b"hello world"
        key = store.save("a/b.txt", io.BytesIO(data), size_hint=len(data))
        self.assertEqual(key, "a/b.txt")
        self.assertTrue(store.exists(key))
        with store.open(key) as fh:
            self.assertEqual(fh.read(), data)
        store.delete(key)
        self.assertFalse(store.exists(key))

    def test_build_local_when_no_bucket(self):
        import config

        config.UPLOAD_BACKEND = "auto"
        config.S3_UPLOAD_BUCKET = ""
        reset_upload_store()
        self.assertIsInstance(build_upload_store(), LocalUploadStore)

    def test_build_unknown_backend_raises(self):
        import config

        config.UPLOAD_BACKEND = "ftp"
        with self.assertRaises(RuntimeError):
            build_upload_store()
        config.UPLOAD_BACKEND = "local"

    def test_get_upload_store_is_cached(self):
        reset_upload_store()
        first = get_upload_store()
        second = get_upload_store()
        self.assertIs(first, second)
        reset_upload_store()


class _FakeS3Client:
    def __init__(self):
        self.objects = {}
        self.upload_calls = []
        self.download_calls = []
        self.head_calls = []
        self.delete_calls = []

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
        fileobj.seek(0)
        self.objects[(bucket, key)] = fileobj.read()
        self.upload_calls.append((bucket, key, ExtraArgs))

    def download_fileobj(self, bucket, key, buf):
        buf.write(self.objects[(bucket, key)])
        buf.seek(0)
        self.download_calls.append((bucket, key))

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise RuntimeError("not found")
        self.head_calls.append((Bucket, Key))

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)
        self.delete_calls.append((Bucket, Key))


class TestS3Storage(unittest.TestCase):
    def setUp(self):
        import config

        self._saved = {
            "UPLOAD_BACKEND": config.UPLOAD_BACKEND,
            "S3_UPLOAD_BUCKET": config.S3_UPLOAD_BUCKET,
            "S3_UPLOAD_PREFIX": config.S3_UPLOAD_PREFIX,
        }
        config.UPLOAD_BACKEND = "s3"
        config.S3_UPLOAD_BUCKET = "my-bucket"
        config.S3_UPLOAD_PREFIX = "uploads"
        # Reset the cached client.
        storage.S3UploadStore._client = None
        self._client = _FakeS3Client()

        def factory(*args, **kwargs):
            return self._client

        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.client = lambda *a, **k: factory()
        sys.modules["boto3"] = fake_boto3
        reset_upload_store()

    def tearDown(self):
        import config

        for k, v in self._saved.items():
            setattr(config, k, v)
        storage.S3UploadStore._client = None
        sys.modules.pop("boto3", None)
        reset_upload_store()

    def test_save_open_exists_delete(self):
        store = S3UploadStore()
        data = b"remote bytes"
        key = store.save("doc.txt", io.BytesIO(data), size_hint=len(data))
        self.assertEqual(key, "doc.txt")
        full = "uploads/doc.txt"
        self.assertIn(("my-bucket", full), self._client.objects)
        self.assertTrue(store.exists(key))
        with store.open(key) as fh:
            self.assertEqual(fh.read(), data)
        store.delete(key)
        self.assertFalse(store.exists(key))

    def test_build_s3_when_bucket_set(self):
        self.assertIsInstance(build_upload_store(), S3UploadStore)

    def test_missing_boto3_raises(self):
        sys.modules.pop("boto3", None)
        storage.S3UploadStore._client = None
        store = S3UploadStore()
        with self.assertRaises(RuntimeError):
            store.save("x.txt", io.BytesIO(b"x"))
        # restore client for tearDown
        storage.S3UploadStore._client = self._client


if __name__ == "__main__":
    unittest.main(verbosity=2)
