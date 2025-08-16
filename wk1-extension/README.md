# TabGenie Chrome Extension

TabGenie is a Chrome extension designed to help you manage your browser tabs effectively. It provides two core functionalities:

## Features

### 1. Organize Tabs

This feature groups similar tabs together based on the similarity of their page titles. It helps in decluttering your browser window by bringing related tabs into proximity.

### 2. Jumble Tabs

This feature randomly shuffles all unpinned tabs in the current browser window. This can be useful for a fresh perspective or simply to add an element of surprise to your browsing.

## How to Install

1.  **Download/Clone:** Obtain the extension files by downloading or cloning this repository.
2.  **Open Chrome Extensions:** Open your Chrome browser and navigate to `chrome://extensions`.
3.  **Enable Developer Mode:** In the top right corner, toggle on "Developer mode".
4.  **Load Unpacked:** Click the "Load unpacked" button.
5.  **Select Extension Folder:** Navigate to and select the `wk1-extension` directory from where you downloaded/cloned the files.
6.  **Icons (Important!):** You will need to manually place actual PNG image files named `icon16.png`, `icon48.png`, and `icon128.png` inside the `wk1-extension/images` directory. Placeholder files created by the setup will cause errors.

## Usage

1.  Click on the TabGenie extension icon in your Chrome toolbar.
2.  In the popup, click either "Organize Tabs" to group similar tabs or "Jumble Tabs" to randomly shuffle them.

## Styling

The popup interface dynamically adjusts its background and text colors based on your browser's dark or light mode settings, providing an aesthetically pleasing experience.

## Key Functions Overview

*   `organizeTabsButton.addEventListener`: Handles the logic for grouping similar tabs based on page title similarity using Jaccard Similarity with N-grams.
*   `jumbleTabsButton.addEventListener`: Handles the logic for randomly shuffling all unpinned tabs in the current window.
*   `jaccardSimilarity(text1, text2)`: Computes the similarity between two tab titles based on their character n-grams.
*   `getNgrams(text, n)`: Helper function to generate character n-grams from a given text.
*   `shuffleArray(array)`: Randomly shuffles the elements of an array using a seeded pseudo-random number generator.
