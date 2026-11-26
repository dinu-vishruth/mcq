import { test } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import React from 'react';
import { create, act } from 'react-test-renderer';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const resource = { id: 17, title: 'Saved aptitude notes', indexed: true, status: 'ready', topic_count: 6 };
const bundle = await build({
  entryPoints: ['src/screens/MakeQuiz.tsx'], bundle: true, write: false,
  platform: 'node', format: 'cjs', jsx: 'automatic', external: ['react', 'react/jsx-runtime'],
  plugins: [{ name: 'ui-boundaries', setup(build) {
    build.onResolve({ filter: /^@\// }, args => ({ path: args.path, namespace: 'stub' }));
    build.onLoad({ filter: /.*/, namespace: 'stub' }, ({ path }) => ({ contents: path === '@/bootstrap' ? `
      export const csrfToken=()=>"csrf";
      export const apiGet=(path)=>globalThis.__quizGet(path);
      export const apiSend=(...args)=>globalThis.__quizSend(...args);
    ` : `import React from 'react';
      const Wrapper=({children})=>React.createElement('div',null,children);
      export const Card=Wrapper,AppShell=Wrapper;
      export const Button=({children,onClick,disabled})=>React.createElement('button',{onClick,disabled},children);` }));
  } }],
});
const mod = { exports: {} };
new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(require, mod, mod.exports);

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

async function renderQuiz(t, generate, options = {}) {
  const submissions = [];
  globalThis.__quizGet = (path) => {
    if (path === '/api/generation/status') return Promise.resolve({ available: true });
    if (path === '/api/knowledge') return options.library ?? Promise.resolve({ items: [resource] });
    return options.job(path);
  };
  globalThis.__quizSend = generate;
  globalThis.window = { location: { search: '?doc=17' } };
  globalThis.document = {
    createElement(tag) {
      if (tag === 'input') return {};
      assert.equal(tag, 'form');
      const fields = new Map();
      return {
        appendChild(input) { fields.set(input.name, input.value); },
        submit() { submissions.push({ method: this.method, action: this.action, fields }); },
      };
    },
    body: { appendChild() {} },
  };
  let root;
  t.after(async () => {
    if (root) await act(async () => root.unmount());
    delete globalThis.window;
    delete globalThis.document;
    delete globalThis.__quizGet;
    delete globalThis.__quizSend;
  });
  await act(async () => {
    root = create(React.createElement(mod.exports.default, { data: {
      username: 'learner', document_id: '17', selected_document: resource,
    } }));
  });
  return {
    root,
    visible: () => JSON.stringify(root.toJSON()),
    button: (label) => root.root.findAllByType('button').find(node => node.children.includes(label)),
    submissions,
  };
}

test('Library link selects a numeric resource and enters a quiz from a legacy response', async (t) => {
  const library = deferred();
  let request;
  const page = await renderQuiz(t, async (path, method, body) => {
    request = { path, method, body };
    return { session_key: 'abcd1234' };
  }, { library: library.promise });
  assert.ok(page.visible().includes('Saved aptitude notes'), 'selected title visible before library fetch');
  await act(async () => library.resolve({ items: [resource] }));
  const selected = page.root.root.findAllByType('button').find(node => node.props['aria-pressed'] === true);
  assert.ok(selected, 'string bootstrap ID is normalized so saved resource is visibly selected');
  const generate = page.button('Generate & Start');
  assert.equal(generate.props.disabled, false);
  await act(async () => generate.props.onClick());
  assert.equal(request.path, '/api/practice/generate');
  assert.equal(request.body.document_id, 17);
  assert.deepEqual(page.submissions, [{ method: 'POST', action: '/student_login',
    fields: new Map([['csrf_token', 'csrf'], ['session_key', 'abcd1234']]) }]);
});

test('inline generation shows activity, then requires consent to start a shorter quiz', async (t) => {
  const result = deferred();
  const page = await renderQuiz(t, () => result.promise);
  let generation;
  await act(async () => { generation = page.button('Generate & Start').props.onClick(); });
  assert.match(page.visible(), /Generating and checking questions/);
  assert.doesNotMatch(page.visible(), /passed the checks so far/);
  await act(async () => {
    result.resolve({ session_key: 'partial', count: 6, requested_count: 10, partial: true,
      quality_policy: 'verified_only', fallback_attempted: true, fallback_count: 0 });
    await generation;
  });
  assert.match(page.visible(), /A shorter quiz is ready/);
  const statuses = page.root.root.findAllByProps({ role: 'status' });
  const count = statuses[0].findAllByType('p')[1].children.join('');
  assert.equal(count, '6 of 10 requested questions passed the checks.');
  assert.match(page.visible(), /A second generation method did not add any questions/);
  assert.equal(page.submissions.length, 0);
  await act(async () => page.button('Start quiz').props.onClick());
  assert.equal(page.submissions[0].fields.get('session_key'), 'partial');
});

test('queued generation reports accepted counts and discloses recovered questions', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const jobs = [
    { status: 'running', progress: 4, total: 10 },
    { status: 'done', result: { session_key: 'recovered', count: 10, requested_count: 10,
      quality_policy: 'verified_only', fallback_attempted: true, fallback_count: 2 } },
  ];
  const page = await renderQuiz(t, async () => ({ job_id: 'job123' }), {
    job: async (path) => {
      assert.equal(path, '/api/jobs/job123');
      return jobs.shift();
    },
  });
  let generation;
  await act(async () => { generation = page.button('Generate & Start').props.onClick(); });
  assert.match(page.visible(), /Waiting to start quiz generation/);
  await act(async () => t.mock.timers.tick(1500));
  assert.match(page.visible(), /4 of 10 requested questions have passed the checks so far/);
  await act(async () => {
    t.mock.timers.tick(1500);
    await generation;
  });
  assert.match(page.visible(), /Your quiz is ready/);
  const count = page.root.root.findAllByProps({ role: 'status' })[0].findAllByType('p')[2].children.join('');
  assert.equal(count, 'A second generation method added 2 questions. These passed the same checks.');
  assert.equal(page.submissions.length, 0);
  await act(async () => page.button('Start quiz').props.onClick());
  assert.equal(page.submissions[0].fields.get('session_key'), 'recovered');
});
