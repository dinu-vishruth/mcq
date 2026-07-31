/* Auth page interactions: password toggle, animated stat counters,
   toast auto-dismiss, and OAuth placeholder feedback. Progressive
   enhancement — the form works with JS disabled. */
(function () {
  'use strict';

  var reduceMotion = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var EYE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>';
  var EYE_OFF = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M9.9 5.2A9.5 9.5 0 0 1 12 5c6.5 0 10 7 10 7a16 16 0 0 1-3 3.7M6.2 6.3A16 16 0 0 0 2 12s3.5 7 10 7a9.5 9.5 0 0 0 4-.9"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/><path d="m3 3 18 18"/></svg>';

  /* ---- Password visibility toggle ---- */
  document.querySelectorAll('[data-pw-toggle]').forEach(function (btn) {
    var input = document.getElementById(btn.getAttribute('data-pw-toggle'));
    if (!input) return;
    btn.addEventListener('click', function () {
      var show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      input.classList.toggle('pw', show);
      btn.innerHTML = show ? EYE_OFF : EYE;
      btn.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
      input.focus();
    });
  });

  /* ---- Toast dismiss (manual + auto) ---- */
  document.querySelectorAll('.toast').forEach(function (toast) {
    var closeBtn = toast.querySelector('[data-toast-close]');
    function dismiss() {
      toast.classList.add('leaving');
      setTimeout(function () { toast.remove(); }, 320);
    }
    if (closeBtn) closeBtn.addEventListener('click', dismiss);
    setTimeout(dismiss, 6000);
  });

  /* ---- OAuth placeholder ---- */
  document.querySelectorAll('[data-oauth]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      showToast(btn.getAttribute('data-oauth') + ' sign-in isn\'t available yet.');
    });
  });

  function showToast(msg) {
    var stack = document.getElementById('toast-stack');
    if (!stack) {
      stack = document.createElement('div');
      stack.className = 'toast-stack';
      stack.id = 'toast-stack';
      document.body.appendChild(stack);
    }
    var el = document.createElement('div');
    el.className = 'toast';
    el.setAttribute('role', 'status');
    el.innerHTML =
      '<span class="ti"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg></span>' +
      '<span></span>' +
      '<button type="button" class="tclose" aria-label="Dismiss"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button>';
    el.querySelector('span:nth-child(2)').textContent = msg;
    function dismiss() { el.classList.add('leaving'); setTimeout(function () { el.remove(); }, 320); }
    el.querySelector('.tclose').addEventListener('click', dismiss);
    stack.appendChild(el);
    setTimeout(dismiss, 6000);
  }

  /* ---- Animated stat counters ---- */
  function formatValue(el, value) {
    var div = parseFloat(el.getAttribute('data-div')) || 1;
    var dec = parseInt(el.getAttribute('data-dec'), 10) || 0;
    var suffix = el.getAttribute('data-suffix') || '';
    var shown = value / div;
    return (dec ? shown.toFixed(dec) : Math.round(shown).toString()) + suffix;
  }

  function runCounter(el) {
    var target = parseFloat(el.getAttribute('data-count')) || 0;
    if (reduceMotion) { el.textContent = formatValue(el, target); return; }
    var duration = 1400, start = null;
    function step(ts) {
      if (start === null) start = ts;
      var p = Math.min((ts - start) / duration, 1);
      var eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
      el.textContent = formatValue(el, target * eased);
      if (p < 1) requestAnimationFrame(step);
      else el.textContent = formatValue(el, target);
    }
    requestAnimationFrame(step);
  }

  var counters = document.querySelectorAll('.authx-stat .num[data-count]');
  if (counters.length) {
    if ('IntersectionObserver' in window) {
      var obs = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { runCounter(e.target); obs.unobserve(e.target); }
        });
      }, { threshold: 0.4 });
      counters.forEach(function (c) { obs.observe(c); });
    } else {
      counters.forEach(runCounter);
    }
  }
})();
