import{c as f}from"./chunk-cn.js";/**
 * @license lucide-react v0.417.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const x=f("Boxes",[["path",{d:"M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z",key:"lc1i9w"}],["path",{d:"m7 16.5-4.74-2.85",key:"1o9zyk"}],["path",{d:"m7 16.5 5-3",key:"va8pkn"}],["path",{d:"M7 16.5v5.17",key:"jnp8gn"}],["path",{d:"M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z",key:"8zsnat"}],["path",{d:"m17 16.5-5-3",key:"8arw3v"}],["path",{d:"m17 16.5 4.74-2.85",key:"8rfmw"}],["path",{d:"M17 16.5v5.17",key:"k6z78m"}],["path",{d:"M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z",key:"1xygjf"}],["path",{d:"M12 8 7.26 5.15",key:"1vbdud"}],["path",{d:"m12 8 4.74-2.85",key:"3rx089"}],["path",{d:"M12 13.5V8",key:"1io7kd"}]]);/**
 * @license lucide-react v0.417.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const M=f("Database",[["ellipse",{cx:"12",cy:"5",rx:"9",ry:"3",key:"msslwz"}],["path",{d:"M3 5V19A9 3 0 0 0 21 19V5",key:"1wlel7"}],["path",{d:"M3 12A9 3 0 0 0 21 12",key:"mv7ke4"}]]);/**
 * @license lucide-react v0.417.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const g=f("FileText",[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M10 9H8",key:"b1mrlr"}],["path",{d:"M16 13H8",key:"t4e002"}],["path",{d:"M16 17H8",key:"z1uh3a"}]]),u=4.5*1024*1024,h=50*1024*1024;function v(){typeof window<"u"&&window.pdfjsLib&&(window.pdfjsLib.GlobalWorkerOptions.workerSrc="/static/js/pdf.worker.min.js")}async function y(e){return new Promise((t,n)=>{const r=new FileReader;r.onload=o=>{var i;return t((i=o.target)==null?void 0:i.result)},r.onerror=n,r.readAsText(e)})}async function k(e){const t=await e.arrayBuffer(),n=await window.pdfjsLib.getDocument({data:t}).promise;let r="";for(let o=1;o<=n.numPages;o++){const p=await(await n.getPage(o)).getTextContent();r+=p.items.map(c=>c.str).join(" ")+`
`}return r}async function m(e){const t=await e.arrayBuffer();return(await window.mammoth.extractRawText({arrayBuffer:t})).value}async function b(e){var n;const t=(n=e.name.split(".").pop())==null?void 0:n.toLowerCase();return t==="txt"?y(e):t==="pdf"?k(e):t==="docx"?m(e):null}function z(e,t=15e4){if(e.length<=t)return e;const n=30,r=Math.floor(t/n),o=e.length/n,i=[];for(let p=0;p<n;p++){const c=Math.floor(p*o);let a=e.indexOf(`
`,c);(a===-1||a>c+1e3)&&(a=e.indexOf(" ",c)),a===-1||a>=e.length?a=c:a+=1;let d=a+r;d>e.length&&(d=e.length);let s=e.indexOf(`
`,d);(s===-1||s>d+1e3)&&(s=e.indexOf(" ",d)),(s===-1||s>e.length)&&(s=d);const l=e.substring(a,s).trim();l.length&&i.push(l)}return i.join(`

... [section transition] ...

`)}function P(e){var n;const t=(n=e.name.split(".").pop())==null?void 0:n.toLowerCase();return t==="pptx"&&e.size>u?"PowerPoint (.pptx) files are limited to 4.5 MB because they're processed on the server. Convert to PDF or compress it.":t!=="pptx"&&e.size>h?"File is too large. Maximum supported size is 50 MB.":""}export{x as B,M as D,g as F,u as M,b as e,v as i,z as s,P as v};
