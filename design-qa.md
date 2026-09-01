# Responsive Radar Email — Design QA

## Comparison target

- Source visual truth: `/tmp/codex-remote-attachments/01a05816-851c-7733-8f7b-83cd18702e32/77F54C75-2996-4BE4-B884-0021F62E37FB/1-照片-1.jpg`
- Source pixels: 1280 × 369 (desktop received-email screenshot).
- Implementation desktop capture: `/Users/eric/.codex/visualizations/2026/09/01/gcn-email-responsive/email-desktop-1440-final.png`
- Implementation mobile capture: `/Users/eric/.codex/visualizations/2026/09/01/gcn-email-responsive/email-mobile-390-final.png`
- Viewports: desktop 1440 × 900 CSS px; mobile 390 × 844 CSS px.
- State: one radar hit, plus empty Hong Kong and A-share summary cards. The source uses eight hits, so this is a layout/reading-order comparison rather than pixel-identical data comparison.
- Density: source and implementation captures were inspected at their native pixel dimensions; no density normalization was required for the layout review.

## Full-view and focused comparison

The source shows a full-width, minimally styled table whose five columns separate too far on desktop and cannot retain a readable hierarchy on a phone. The implementation constrains the report to a 680px email card, provides a clear scan header, groups market totals into cards, and keeps the desktop table compact.

Focused review of the mobile signal area confirmed that at 390px the table header is hidden, each `.signal-row` becomes a card, labels are visible, and the document width remains exactly 390px. At desktop width the row remains a table row and the mobile-only labels are hidden.

## Required fidelity surfaces

- Fonts and typography: System email-safe stack; one strong title level, compact section headings, 11–14px supporting text, and readable signal pills. No truncation in the tested row.
- Spacing and layout rhythm: 24px desktop card padding reduces to 16px on mobile; summary and signal sections use distinct, consistent gaps. The fixed 680px desktop card avoids the source's excessive column spread.
- Colors and visual tokens: Restrained navy header, white content card, slate metadata, and semantic red/amber signal pills; contrast is sufficient against white backgrounds.
- Image quality and asset fidelity: The report uses no image, logo, or decorative asset; none was substituted or omitted.
- Copy and content: Market scan, hit, failure, status, symbol, name, close, signal, date, and the existing EHOPT10 disclaimer remain present. The mobile variant adds field labels without changing data.

## Findings and iteration history

- [P1, fixed] The original layout allowed the signal table to spread across the mail viewport, making relationships between name, close, and signal difficult to scan. Fixed with a constrained email shell, summary cards, and a responsive row-card breakpoint.
- [P1, fixed] First implementation preview exposed mobile field labels in the desktop table. Added the default `.cell-label{display:none!important}` rule; the 1440px capture now measures `display: none`, while the 390px capture measures `inline-block`.

## Validation evidence

- Desktop: 680px card centered in a 1440px viewport; document `scrollWidth` equals 1440px; table header is `table-header-group`; signal row is `table-row`.
- Mobile: document `scrollWidth` equals 390px; table header is hidden; signal row is `block`; no browser console warnings or errors.
- Automated regression: `python3 tests/run_all.py` passed 123/123 after adding responsive-email coverage.

## Implementation checklist

- [x] Preserve plain-text multipart fallback.
- [x] Keep all report data in the HTML version.
- [x] Add compact desktop summary and signal hierarchy.
- [x] Stack summary cards and signal fields below 600px.
- [x] Verify desktop and mobile rendering states.

## Follow-up polish

- [P3] Consider an optional direct link to the radar workspace if a stable public HTTPS URL is introduced.

final result: passed

# Nine-turn Indicator Markers — Design QA

## Comparison target

- Source visual truth: `/var/folders/q7/zx3ty_wj5116zxv1gtv8djbh0000gn/T/codex-clipboard-3e7a7e22-9e13-4b9d-b2d4-78fbbd029797.png` (Futu TQQQ chart).
- Local implementation capture: `/Users/eric/.codex/visualizations/2026/09/01/gcn-nine-turn-qa/implementation-2048x1209.png`.
- Combined review image: `/Users/eric/.codex/visualizations/2026/09/01/gcn-nine-turn-qa/reference-vs-implementation.png`.
- Viewport: 2048 × 1209 CSS px; TQQQ daily K-line, v4, 2,500 bars.

## Findings and fixes

- [P1, fixed] The iterative implementation displayed every causal 1–8 partial sequence, producing many isolated numbers that do not match the reference indicator. Restored the original display contract: backfill 1–8 only when a sequence completes at 9, or display the current final-bar sequence when it has reached 5–8.
- [P1, fixed] Completion points existed in the payload and signal history but were omitted from the chart. Added upper-nine and lower-nine chart series using the original reference colors: upper 9 green, lower 9 magenta.
- [P1, fixed] The generic ECharts collision policy hid alternating digits inside valid completed sequences, so compact runs appeared as 1/3/5/7/9. Nine-turn series now opt out of `hideOverlap`, preserving every 1–9 label while other marker series retain collision protection.
- The raw consecutive counts remain unchanged because they are used by the existing strategy and tooltip. Only the visual label mask changed.

## Visual verification

- Completed TQQQ sequences now render as compact 1–9 groups at the same turning regions visible in the reference, including the 2026-03-20 lower-nine, 2026-04-14 upper-nine, 2026-06-03 upper-nine, and 2026-07-27 lower-nine completions.
- Incomplete short runs no longer scatter 1–4 labels across the chart.
- Every digit in the completed 1–9 runs remains visible at the tested 2048 × 1201 viewport; the previously missing 2/4/6/8 labels are present.
- The external `九转` legend still toggles all 1–9 label series as one group.
- Browser console warnings/errors: none.

## Regression evidence

- The v3 and v4 nine-turn label columns are again included in full golden-output comparison.
- `python3 tests/run_all.py`: 130/130 passed.
- `python3 -m py_compile gcn/recipes/gcn_main.py tests/test_recipe.py tests/test_golden.py tests/test_webui.py`: passed.
- `git diff --check`: passed.

Latest comparison evidence:

- `/Users/eric/.codex/visualizations/2026/09/01/gcn-nine-turn-qa/all-digits-2048x1201.png`
- `/Users/eric/.codex/visualizations/2026/09/01/gcn-nine-turn-qa/reference-vs-all-digits-final.png`

final result: passed
