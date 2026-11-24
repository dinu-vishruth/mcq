import{c as p}from"./chunk-cn-C96feRYP.js";/**
 * @license lucide-react v0.417.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const y=p("FileText",[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M10 9H8",key:"b1mrlr"}],["path",{d:"M16 13H8",key:"t4e002"}],["path",{d:"M16 17H8",key:"z1uh3a"}]]),l=4.5*1024*1024,w=50*1024*1024;function M(){typeof window<"u"&&window.pdfjsLib&&(window.pdfjsLib.GlobalWorkerOptions.workerSrc="/static/js/pdf.worker.min.js")}async function h(e){return new Promise((t,n)=>{const o=new FileReader;o.onload=i=>{var s;return t((s=i.target)==null?void 0:s.result)},o.onerror=n,o.readAsText(e)})}async function m(e){const t=await e.arrayBuffer(),n=await window.pdfjsLib.getDocument({data:t}).promise;let o="";for(let i=1;i<=n.numPages;i++){const u=await(await n.getPage(i)).getTextContent();o+=u.items.map(c=>c.str).join(" ")+`
`}return o}async function x(e){const t=await e.arrayBuffer();return(await window.mammoth.extractRawText({arrayBuffer:t})).value}async function k(e){var n;const t=(n=e.name.split(".").pop())==null?void 0:n.toLowerCase();return t==="txt"?h(e):t==="pdf"?m(e):t==="docx"?x(e):null}function b(e,t=15e4){if(e.length<=t)return e;const n=30,o=Math.floor(t/n),i=e.length/n,s=[];for(let u=0;u<n;u++){const c=Math.floor(u*i);let r=e.indexOf(`
`,c);(r===-1||r>c+1e3)&&(r=e.indexOf(" ",c)),r===-1||r>=e.length?r=c:r+=1;let f=r+o;f>e.length&&(f=e.length);let a=e.indexOf(`
`,f);(a===-1||a>f+1e3)&&(a=e.indexOf(" ",f)),(a===-1||a>e.length)&&(a=f);const d=e.substring(r,a).trim();d.length&&s.push(d)}return s.join(`

... [section transition] ...

`)}function P(e){var n;const t=(n=e.name.split(".").pop())==null?void 0:n.toLowerCase();return t==="pptx"&&e.size>l?"PowerPoint (.pptx) files are limited to 4.5 MB because they're processed on the server. Convert to PDF or compress it.":t!=="pptx"&&e.size>w?"File is too large. Maximum supported size is 50 MB.":""}export{y as F,l as M,k as e,M as i,b as s,P as v};
