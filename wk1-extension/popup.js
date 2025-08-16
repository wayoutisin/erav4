document.addEventListener('DOMContentLoaded', () => {
  const organizeTabsButton = document.getElementById('organizeTabs');
  const jumbleTabsButton = document.getElementById('jumbleTabs');

  // Helper to generate n-grams from a string
  const getNgrams = (text, n = 3) => {
    if (!text) return new Set();
    const normalizedText = text.toLowerCase().replace(/[^a-z0-9]/g, '');
    const ngrams = new Set();
    for (let i = 0; i <= normalizedText.length - n; i++) {
      ngrams.add(normalizedText.substring(i, i + n));
    }
    return ngrams;
  };

  // Jaccard Similarity based on n-grams
  const jaccardSimilarity = (text1, text2) => {
    const ngrams1 = getNgrams(text1);
    const ngrams2 = getNgrams(text2);

    if (ngrams1.size === 0 || ngrams2.size === 0) return 0;

    const intersection = new Set([...ngrams1].filter(x => ngrams2.has(x)));
    const union = new Set([...ngrams1, ...ngrams2]);

    return intersection.size / union.size;
  };

  // Simple PRNG for deterministic shuffling
  function mulberry32(a) {
    return function() {
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t = t ^ t << 7;
      t = t ^ t >>> 14;
      return ((t >>> 0) / 4294967296);
    }
  }

  let seededRandom = mulberry32(Date.now()); // Initialize with current time as seed

  function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(seededRandom() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  organizeTabsButton.addEventListener('click', () => {
    chrome.tabs.query({}, (tabs) => {
      const groupedTabs = [];
      const processedTabIds = new Set();

      tabs.forEach(tab1 => {
        if (!processedTabIds.has(tab1.id)) {
          const similarTabs = [tab1];
          processedTabIds.add(tab1.id);

          tabs.forEach(tab2 => {
            if (!processedTabIds.has(tab2.id) && tab1.id !== tab2.id) {
              // Use Jaccard similarity for comparison
              if (jaccardSimilarity(tab1.title, tab2.title) > 0.2) { // Adjusted threshold for Jaccard
                similarTabs.push(tab2);
                processedTabIds.add(tab2.id);
              }
            }
          });
          groupedTabs.push(similarTabs);
        }
      });

      // Flatten groupedTabs for processing, maintaining order within groups
      const reorderedTabs = [];
      groupedTabs.forEach(group => {
        reorderedTabs.push(...group);
      });

      // Now, move the tabs based on reorderedTabs
      let index = 0;
      reorderedTabs.forEach(tab => {
        chrome.tabs.move(tab.id, { index: index++ });
      });
    });
  });

  jumbleTabsButton.addEventListener('click', () => {
    chrome.tabs.query({ currentWindow: true }, (tabs) => {
      const movableTabs = tabs.filter(tab => !tab.pinned); // Don't jumble pinned tabs
      const shuffledTabs = shuffleArray([...movableTabs]);

      shuffledTabs.forEach((tab, index) => {
        chrome.tabs.move(tab.id, { index: index });
      });
    });
  });
});
