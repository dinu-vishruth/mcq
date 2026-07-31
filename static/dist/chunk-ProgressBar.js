import{c as i,m as r,a as e}from"./chunk-cn.js";import{j as a}from"./app.js";/**
 * @license lucide-react v0.417.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const m=i("Clock",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["polyline",{points:"12 6 12 12 16 14",key:"68esgv"}]]);function u({pct:c,tone:s="accent",className:t}){const n=Math.max(0,Math.min(100,c)),o={accent:"bg-accent",danger:"bg-gradient-to-r from-danger to-[#f87171]",success:"bg-success"}[s];return a.jsx("div",{className:e("h-2 rounded-full bg-white/[0.06] overflow-hidden",t),children:a.jsx(r.div,{className:e("h-full rounded-full",o),initial:{width:0},animate:{width:`${n}%`},transition:{duration:.7,ease:[.16,1,.3,1]}})})}export{m as C,u as P};
