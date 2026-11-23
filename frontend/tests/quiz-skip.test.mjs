import { test } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import React from 'react';
import { create, act } from 'react-test-renderer';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

const bundle = await build({
  entryPoints: ['src/screens/Quiz.tsx'], bundle: true, write: false,
  platform: 'node', format: 'cjs', jsx: 'automatic', external: ['react', 'react/jsx-runtime'],
  plugins: [{ name: 'presentation-boundaries', setup(build) {
    build.onResolve({ filter: /^@\/|^framer-motion$/ }, args => ({ path: args.path, namespace: 'stub' }));
    build.onLoad({ filter: /.*/, namespace: 'stub' }, ({ path }) => ({ contents:
      path === '@/bootstrap' ? 'export const csrfToken=()=>"csrf"; export const apiSend=async()=>({});' : `
        import React from 'react';
        const Wrapper=({children})=>React.createElement('div',null,children);
        export const AnimatePresence=Wrapper, ProgressBar=Wrapper;
        export const motion={div:Wrapper,span:Wrapper};
        export const Button=({children,onClick,disabled})=>React.createElement('button',{onClick,disabled},children);`
    }));
  } }],
});
const mod = { exports: {} };
new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(require, mod, mod.exports);

async function quiz(total = 3, timer = 600) {
  let root, submitted, submissionCount = 0;
  const spent = { value: '0' };
  const form = { elements: { namedItem: () => spent }, submit() {
    submissionCount++;
    submitted = Object.fromEntries(root.root.findAllByType('input').map(n => [n.props.name, n.props.value ?? n.props.defaultValue]));
    submitted.time_spent = spent.value;
  } };
  await act(async () => { root = create(React.createElement(mod.exports.default, { data: {
    username: 'learner', timer, deadline: 1000 + timer, server_now: 1000, session_key: 'quiz-test', user_id: 1, answers: {}, confidence: {}, question_index: 0, mcqs: Array.from({ length: total }, (_, i) => ({
      question: `Question text ${i + 1}`, answer_text: 'Correct', options: [
        { label: 'A', text: 'Correct' }, { label: 'B', text: 'Other' },
      ],
    })),
  } }), { createNodeMock: node => node.type === 'form' ? form : {} }); });
  async function click(label) {
    const button = root.root.findAllByType('button').find(n => n.children.includes(label));
    assert.ok(button, `button ${label} exists`);
    assert.ok(!button.props.disabled, `button ${label} enabled`);
    await act(async () => button.props.onClick());
  }
  return { root, click, submitted: () => submitted, submissionCount: () => submissionCount,
    close: () => act(async () => root.unmount()) };
}

test('skip advances, clears a selected answer, and preserves answers to other questions', async () => {
  const q = await quiz();
  try {
    await q.click('Skip');
    assert.ok(JSON.stringify(q.root.toJSON()).includes('Question text 2'));
    await q.click('Previous');
    const option = q.root.root.findAllByType('button').find(n => n.children.some(c => c?.props?.children === 'Correct'));
    await act(async () => option.props.onClick());
    await q.click('Skip');
    assert.equal(q.root.root.findByProps({ name: 'q-0' }).props.value, '');
    const answer = q.root.root.findAllByType('button').find(n => n.children.some(c => c?.props?.children === 'Other'));
    await act(async () => answer.props.onClick());
    await q.click('Next'); await q.click('Skip');
    assert.equal(q.root.root.findByProps({ role: 'status' }).children.join(''), '1 answered · 2 unanswered');
    await q.click('Review unanswered');
    assert.ok(JSON.stringify(q.root.toJSON()).includes('Question text 1'));
    assert.equal(q.root.root.findByProps({ name: 'q-1' }).props.value, 'Other');
  } finally { await q.close(); }
});

test('skipping every question allows submission with empty answers', async () => {
  const q = await quiz();
  try {
    await q.click('Skip'); await q.click('Skip'); await q.click('Skip');
    assert.equal(q.submitted(), undefined, 'last skip lets the learner review before submitting');
    await q.click('Submit Answers');
    for (let i = 0; i < 3; i++) assert.equal(q.submitted()[`q-${i}`], '');
  } finally { await q.close(); }
});

test('timer still submits a skipped single-question quiz from the review screen', async () => {
  const q = await quiz(1, 1);
  try {
    await q.click('Skip');
    await act(async () => new Promise(resolve => setTimeout(resolve, 1100)));
    assert.equal(q.submitted()['q-0'], '');
    assert.equal(q.submissionCount(), 1);
  } finally { await q.close(); }
});

test('repeated clicks and timer expiry cannot submit an attempt twice', async () => {
  const q = await quiz(1, 1);
  try {
    await q.click('Skip');
    const button = q.root.root.findAllByType('button').find(n => n.children.includes('Submit Answers'));
    await act(async () => { button.props.onClick(); button.props.onClick(); });
    await act(async () => new Promise(resolve => setTimeout(resolve, 1100)));
    assert.equal(q.submissionCount(), 1);
    const submitting = q.root.root.findAllByType('button').find(n => n.children.includes('Submitting\u2026'));
    assert.equal(submitting.props.disabled, true);
  } finally { await q.close(); }
});
