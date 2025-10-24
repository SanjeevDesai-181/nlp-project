document.addEventListener("DOMContentLoaded", () => {
  // --- Element Selection ---
  const chatWindow = document.getElementById("chat-window");
  const promptForm = document.getElementById("prompt-form-improved"); // Consistent ID from HTML
  const userInput = document.getElementById("user-input");
  const qaMode = document.getElementById("qa-mode");
  const historyList = document.getElementById("history-list");
  const clearHistoryBtn = document.getElementById("clear-history-btn");
  const sendBtn = document.getElementById("send-btn");

  // --- State Management ---
  let currentConversation = [];
  let historyData = JSON.parse(localStorage.getItem("qaHistory")) || [];

  // --- Utility Functions ---

  /**
   * Creates and appends a chat bubble to the chat window.
   * @param {string} sender - 'user' or 'system'
   * @param {string} message - The text content
   * @param {string} mode - The QA mode used (for system messages)
   */
  function createChatBubble(sender, message, mode = null) {
    // Remove initial prompt if it exists
    const initialPrompt = document.querySelector(".initial-prompt");
    if (initialPrompt) {
      initialPrompt.remove();
    }

    const bubble = document.createElement("div");
    bubble.classList.add("chat-bubble", sender);

    let avatarIcon = sender === "user" ? "person" : "psychology"; // Material Icon names

    // Use template literal for clean HTML injection
    bubble.innerHTML = `
            <div class="chat-avatar material-icons">${avatarIcon}</div>
            <div class="chat-content">
                ${mode ? `<span class="mode-tag">${mode}</span>` : ""}
                <p>${message}</p>
            </div>
        `;

    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Auto-scroll to bottom
  }

  /**
   * Stores the current conversation to history.
   * @param {string} question - The user's initial question.
   */
  function saveHistory(question) {
    if (question.trim() === "") return;

    // Create a short title for the history entry
    const historyTitle =
      question.length > 30 ? question.substring(0, 30) + "..." : question;

    // Add a new entry to the history (at the beginning of the array)
    historyData.unshift({
      id: Date.now(),
      question: historyTitle,
      conversation: [...currentConversation],
    });

    // Keep history limited (e.g., last 15 conversations)
    historyData = historyData.slice(0, 15);
    localStorage.setItem("qaHistory", JSON.stringify(historyData));

    loadHistory(); // Reload the sidebar history
  }

  /**
   * Loads and displays history entries in the sidebar.
   */
  function loadHistory() {
    historyList.innerHTML = "";
    if (historyData.length === 0) {
      historyList.innerHTML =
        '<p class="no-history-message">Start a conversation...</p>';
      return;
    }

    historyData.forEach((item, index) => {
      const historyItem = document.createElement("div");
      historyItem.classList.add("history-item");
      historyItem.textContent = item.question;
      historyItem.dataset.index = index;

      historyItem.addEventListener("click", () => {
        loadConversation(item.conversation);
      });
      historyList.appendChild(historyItem);
    });
  }

  /**
   * Clears the chat window and loads a past conversation.
   * @param {Array<Object>} conversation - The array of past chat messages.
   */
  function loadConversation(conversation) {
    chatWindow.innerHTML = ""; // Clear current chat
    currentConversation = [...conversation]; // Set the new conversation
    conversation.forEach((chat) => {
      // Re-render each message
      createChatBubble(chat.sender, chat.message, chat.mode);
    });
  }

  /**
   * Handles the form submission (sending the question to the backend).
   */
  promptForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const question = userInput.value.trim();
    const mode = qaMode.value;
    const modeLabel = qaMode.options[qaMode.selectedIndex].text;

    if (!question) return;

    // 1. Display User Message
    createChatBubble("user", question);
    currentConversation.push({ sender: "user", message: question });

    // 2. Clear Input and Disable Button
    userInput.value = "";
    sendBtn.disabled = true;

    // --- ACTUAL API CALL to Python Backend ---
    fetch("http://127.0.0.1:5000/api/answer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: question, mode: mode }),
    })
      .then((response) => {
        if (!response.ok) {
          // Handle HTTP error statuses (e.g., 400, 500)
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        // 4. Display System Message from API response
        let answer = data.answer || "No response field received from server.";

        // --- OPTIMIZATION: Remove [Mode-Prefix] from answer text ---
        // This assumes the Python backend includes brackets and a space, e.g., "[Rule-Based] "
        if (answer.startsWith("[")) {
          answer = answer.substring(answer.indexOf("]") + 2).trim();
        }
        // -----------------------------------------------------------

        createChatBubble("system", answer, modeLabel);
        currentConversation.push({
          sender: "system",
          message: answer,
          mode: modeLabel,
        });

        // 5. Save History and Re-enable Button
        saveHistory(question);
      })
      .catch((error) => {
        console.error("API Error:", error);
        // Display connection/server error message
        const errorMessage = `[Server Error] Could not reach the Python backend (mode: ${mode}). Please ensure app.py is running.`;
        createChatBubble("system", errorMessage, "Error");
        currentConversation.push({
          sender: "system",
          message: errorMessage,
          mode: "Error",
        });
      })
      .finally(() => {
        sendBtn.disabled = false;
      });
  });

  /**
   * Clears all history from localStorage and updates the UI.
   */
  clearHistoryBtn.addEventListener("click", () => {
    if (confirm("Are you sure you want to clear all chat history?")) {
      localStorage.removeItem("qaHistory");
      historyData = [];
      loadHistory(); // Update the sidebar
      chatWindow.innerHTML = `
                <div class="initial-prompt">
                    <span class="material-icons gemini-icon">psychology</span>
                    <h1>History Cleared. Start a new conversation.</h1>
                    <p>I can answer questions like "Who is his father?" (Rule-Based) or "Summarize his T20I career." (Generative).</p>
                </div>
            `;
      currentConversation = []; // Clear current session state
    }
  });

  // --- Initial Load ---
  loadHistory();
});
