var codeBlocks = document.querySelectorAll('div.highlight');

codeBlocks.forEach(function (codeBlock) {
    var copyButton = document.createElement('button');
    // copyButton.className = 'copy';
    copyButton.type = 'button';
    copyButton.ariaLabel = 'Copy code to clipboard';
    copyButton.innerText = '📋 Copy code';

    var languageSpan = document.createElement('span');
    languageSpan.className = 'language-name';
    languageSpan.textContent = codeBlock.parentElement.classList[0].replace('language-', '');

    var copyCodeDiv = document.createElement('div');
    copyCodeDiv.className = 'copy-code';
    copyCodeDiv.append(languageSpan);
    copyCodeDiv.append(copyButton);

    codeBlock.prepend(copyCodeDiv);

    copyButton.addEventListener('click', function () {
      var code = codeBlock.querySelector('code').innerText.trim();
      window.navigator.clipboard.writeText(code);

      copyButton.innerText = '✅ Copied!';

      setTimeout(function () {
        copyButton.innerText = '📋 Copy code';
      }, 2000);
    });
});
