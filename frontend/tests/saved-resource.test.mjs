import { test } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import React from 'react';
import { create, act } from 'react-test-renderer';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

test('Library link selects a numeric resource and generates without an upload', async () => {
  const resource = { id: 17, title: 'Saved aptitude notes', indexed: true, status: 'ready', topic_count: 6 };
  let resolveList, request, submitted = false;
  globalThis.__libraryList = () => new Promise(resolve => { resolveList = resolve; });
  globalThis.__generate = async (path, method, body) => {
    request = { path, method, body };
    return { session_key: 'abcd1234' };
  };
  globalThis.window = { location: { search: '?doc=17' } };
  globalThis.document = {
    createElement: () => ({ submit() { submitted = true; } }),
    body: { appendChild() {} },
  };
  const bundle = await build({
    entryPoints: ['src/screens/MakeQuiz.tsx'], bundle: true, write: false,
    platform: 'node', format: 'cjs', jsx: 'automatic', external: ['react', 'react/jsx-runtime'],
    plugins: [{ name: 'ui-boundaries', setup(build) {
      build.onResolve({ filter: /^@\// }, args => ({ path: args.path, namespace: 'stub' }));
      build.onLoad({ filter: /.*/, namespace: 'stub' }, ({ path }) => ({ contents: path === '@/bootstrap' ? `
        export const csrfToken=()=>"csrf";
        export const apiGet=(path,...args)=>path==="/api/generation/status" ? Promise.resolve({available:true,message:"Ready"}) : globalThis.__libraryList(path,...args);
        export const apiSend=(...args)=>globalThis.__generate(...args);
      ` : `import React from 'react';
        const Wrapper=({children})=>React.createElement('div',null,children);
        export const Card=Wrapper,AppShell=Wrapper;
        export const Button=({children,onClick,disabled})=>React.createElement('button',{onClick,disabled},children);` }));
    } }],
  });
  const mod = { exports: {} };
  new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(require, mod, mod.exports);
  let root;
  await act(async () => {
    root = create(React.createElement(mod.exports.default, { data: {
      username: 'learner', document_id: '17', selected_document: resource,
    } }));
  });
  assert.ok(JSON.stringify(root.toJSON()).includes('Saved aptitude notes'), 'selected title visible before library fetch');
  await act(async () => resolveList({ items: [resource] }));
  const selected = root.root.findAllByType('div').find(node => node.props.className?.includes('border-accent bg-accent/8'));
  assert.ok(selected, 'string bootstrap ID is normalized so saved resource is visibly selected');
  const generate = root.root.findAllByType('button').find(node => node.children.includes('Generate & Start'));
  assert.equal(generate.props.disabled, false);
  await act(async () => generate.props.onClick());
  assert.equal(request.path, '/api/practice/generate');
  assert.equal(request.body.document_id, 17);
  assert.equal(submitted, true, 'successful generation enters the quiz');
  await act(async () => root.unmount());
  delete globalThis.window;
  delete globalThis.document;
});
