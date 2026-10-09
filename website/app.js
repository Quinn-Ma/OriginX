'use strict';
const copyButton = document.getElementById('copy-citation');
copyButton?.addEventListener('click', async () => {
  const text = document.getElementById('citation-text').textContent;
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = 'Copied';
    status.textContent = 'BibTeX citation copied to clipboard.';
    setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2000);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('citation-text'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = 'Citation selected';
    status.textContent = 'Clipboard unavailable. The citation is selected; use your copy command.';
  }
});
