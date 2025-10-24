document.addEventListener("DOMContentLoaded", () => {
  const chatWindow = document.getElementById("chat-window");
  const promptForm = document.getElementById("prompt-form");
  const userInput = document.getElementById("user-input");
  const qaMode = document.getElementById("qa-mode");
  const historyList = document.getElementById("history-list");

  // --- State Management ---
  let currentConversation = [];
  let historyData = JSON.parse(localStorage.getItem("qaHistory")) || [];

  // --- Utility Functions ---

  // Function to create and append a chat bubble
  function createChatBubble(sender, message, mode = null) {
    const bubble = document.createElement("div");
    bubble.classList.add("chat-bubble", sender);

    let avatarContent = sender === "user" ? "U" : "AI";
    let avatarStyle = sender === "user" ? "person" : "psychology"; // Material Icon names

    bubble.innerHTML = `
            <div class="chat-content">
                ${mode ? `<span class="mode-tag">${mode}</span>` : ""}
                <p>${message}</p>
            </div>
            <span class="chat-avatar material-icons">${avatarStyle}</span>
        `;

    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Auto-scroll to bottom
  }

  // Function to save the current conversation to history
  function saveHistory(question) {
    if (question.trim() === "") return;

    // Add a new entry to the history
    historyData.unshift({
      id: Date.now(),
      question: question.substring(0, 50) + (question.length > 50 ? "..." : ""),
      conversation: [...currentConversation],
    });

    // Keep history limited (e.g., last 10 conversations)
    historyData = historyData.slice(0, 10);
    localStorage.setItem("qaHistory", JSON.stringify(historyData));

    // Reload the sidebar history
    loadHistory();
  }

  // Function to load and display history in the sidebar
  function loadHistory() {
    historyList.innerHTML = "";
    if (historyData.length === 0) {
      historyList.innerHTML =
        '<p class="no-history">Start a conversation to see history.</p>';
      return;
    }

    historyData.forEach((item, index) => {
      const historyItem = document.createElement("div");
      historyItem.classList.add("history-item");
      historyItem.textContent = item.question;
      historyItem.dataset.index = index;
      historyItem.addEventListener("click", () => {
        // Load clicked history item
        loadConversation(item.conversation);
      });
      historyList.appendChild(historyItem);
    });
  }

  // Function to display a past conversation
  function loadConversation(conversation) {
    chatWindow.innerHTML = ""; // Clear current chat
    currentConversation = [...conversation]; // Set the new conversation
    conversation.forEach((chat) => {
      // Re-render each message
      createChatBubble(chat.sender, chat.message, chat.mode);
    });
  }

  // --- Main Logic (Simulated QA System) ---
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
    document.getElementById("send-btn").disabled = true;

    // 3. Simulate API Call/Processing (Replace this with your actual Python/API call later)
    setTimeout(() => {
      let answer = "";

      // Generate a simulated response based on the selected mode
      if (mode === "rule-based") {
        answer = `[RULE-BASED EXTRACT] The simple algorithm found a keyword match in chunk 15 and extracted the surrounding text: "The main product is the X-900, which has a 5-year warranty and a required maintenance schedule of every 2,000 operational hours."`;
      } else if (mode === "llm-extractive") {
        answer = `[LLM EXTRACTIVE] Based on the top 3 retrieved chunks, the LLM identified the exact sentence that answers your question: "The current procedure for handling errors requires cross-validation steps with the data log, as mandated by Section 4.1.2 of the manual."`;
      } else if (mode === "llm-generative") {
        answer = `[LLM GENERATIVE] The system synthesized information from three separate paragraphs (sections A, B, and C) of the PDF to provide this cohesive answer: The recommended next step is to first verify the system integrity using the /check-status command. If the status returns 'GREEN', the fault is likely electrical, requiring a physical inspection. If the status is 'RED', proceed to the diagnostic section starting on page 45.`;
      } else {
        answer = "Error: Invalid QA Mode selected.";
      }

      // 4. Display System Message
      createChatBubble("system", answer, modeLabel);
      currentConversation.push({
        sender: "system",
        message: answer,
        mode: modeLabel,
      });

      // 5. Save History and Re-enable Button
      saveHistory(question);
      document.getElementById("send-btn").disabled = false;
    }, 1500); // Simulate 1.5 second latency
  });

  // Initial load
  loadHistory();
});
