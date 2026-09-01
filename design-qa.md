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
