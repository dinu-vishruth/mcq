import { test } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import React from 'react';
import { create, act } from 'react-test-renderer';
import { createRequire } from 'node:module';

// Exercise the actual screen state transitions, holding extraction open while
// React commits the busy state. Stub presentation and browser APIs only.
const require = createRequire(import.meta.url);
for (const screen of ['Upload', 'AddResource']) {
  for (const scenario of ['extracted', 'server', 'extraction-error']) {
    test(`${screen}: preserves submission for ${scenario}`, async () => {
      let resolveText, rejectText;
      const extraction = new Promise((resolve, reject) => {
        resolveText = resolve; rejectText = reject;
      });
      globalThis.__extract = () => extraction;
      const result = await build({
        entryPoints: [`src/screens/${screen}.tsx`], bundle: true,
        write: false, platform: 'node', format: 'cjs', jsx: 'automatic',
        external: ['react', 'react/jsx-runtime'],
        plugins: [{ name: 'test-browser-boundaries', setup(build) {
          build.onResolve({ filter: /^@\/|^framer-motion$/ }, args => ({ path: args.path, namespace: 'stub' }));
          build.onLoad({ filter: /.*/, namespace: 'stub' }, ({ path }) => {
            if (path === '@/lib/extract') return { contents: `
              export const MAX_UPLOAD_SIZE=4718592;
              export const initPdfWorker=()=>{};
              export const validateFileSize=()=>null;
              export const sampleText=t=>t;
              export const extractText=f=>globalThis.__extract(f);` };
            if (path === '@/bootstrap') return { contents: 'export const csrfToken=()=>"test-csrf";' };
            return { contents: `import React from 'react';
              const Wrapper=({children})=>React.createElement('div',null,children);
              export const Card=Wrapper, Button=Wrapper, AppShell=Wrapper, AnimatePresence=Wrapper;
              export const motion={span:Wrapper,p:Wrapper};` };
          });
        } }],
      });
      const mod = { exports: {} };
      new Function('require', 'module', 'exports', result.outputFiles[0].text)(require, mod, mod.exports);
      const file = { name: scenario === 'server' ? 'notes.pptx' : 'notes.pdf', size: 100 };
      const input = { files: [file], removeAttribute() {}, set value(v) { if (!v) this.files = []; } };
      const textInput = { value: '' };
      let submitted;
      const form = { submit() { submitted = { files: [...input.files], text: textInput.value }; } };
      let root;
      await act(async () => {
        root = create(React.createElement(mod.exports.default, { data: { username: 'tester' } }), {
          createNodeMock: element => element.type === 'form' ? form :
            element.props.type === 'file' ? input :
            element.props.name === 'extracted_text' ? textInput : {},
        });
      });
      await act(async () => root.root.findByProps({ type: 'file' }).props.onChange({ target: { files: [file] } }));
      let pending;
      await act(async () => { pending = root.root.findByType('form').props.onSubmit({ preventDefault() {} }); });
      assert.equal(root.root.findByProps({ type: 'file' }).instance, input, 'file input survives busy render');
      if (scenario === 'extraction-error') rejectText(new Error('PDF parser unavailable'));
      else resolveText(scenario === 'extracted' ? 'Study material for quiz generation.' : '');
      await act(async () => { await pending; await new Promise(resolve => setTimeout(resolve, 650)); });
      assert.ok(submitted, 'form submitted');
      if (scenario === 'extracted') {
        assert.equal(submitted.text, 'Study material for quiz generation.');
        assert.equal(submitted.files.length, 0, 'omit original file to respect Vercel body limit');
      } else {
        assert.deepEqual(submitted.files, [file], 'server extraction receives original file');
      }
      await act(async () => root.unmount());
    });
  }
}
