/**
 * Example Frontend Integration for RAG Chatbot API
 * Works with React, Vue, or vanilla JavaScript
 */

// ============================================
// Configuration
// ============================================

const API_URL = 'http://localhost:8000';  // Change to your deployed URL

// ============================================
// React/Next.js Example
// ============================================

import { useState } from 'react';

function ChatbotComponent() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message to UI
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Call API
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        credentials: 'include',  // CRITICAL: Include cookies for sessions
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          prompt: input,
          // Optional parameters:
          // temperature: 0.2,
          // k_documents: 3
        })
      });

      const data = await response.json();

      // Add AI response to UI
      const aiMessage = { 
        role: 'assistant', 
        content: data.answer,
        sources: data.sources,
        memory_status: data.memory_status
      };
      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error('Error:', error);
      alert('Failed to send message');
    } finally {
      setLoading(false);
    }
  };

  const clearSession = async () => {
    try {
      await fetch(`${API_URL}/api/clear-session`, {
        method: 'POST',
        credentials: 'include'
      });
      setMessages([]);
      alert('Conversation cleared!');
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="chatbot-container">
      {/* Messages */}
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="content">{msg.content}</div>
            
            {/* Show sources for AI messages */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources">
                <strong>Sources:</strong>
                {msg.sources.map((src, i) => (
                  <div key={i} className="source">
                    Page {src.page}: {src.content}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? 'Sending...' : 'Send'}
        </button>
        <button onClick={clearSession}>Clear</button>
      </div>
    </div>
  );
}

export default ChatbotComponent;


// ============================================
// Vanilla JavaScript Example
// ============================================

class Chatbot {
  constructor(apiUrl, containerId) {
    this.apiUrl = apiUrl;
    this.container = document.getElementById(containerId);
    this.messages = [];
    this.init();
  }

  init() {
    this.container.innerHTML = `
      <div id="messages"></div>
      <div class="input-area">
        <input type="text" id="chat-input" placeholder="Ask a question..." />
        <button id="send-btn">Send</button>
        <button id="clear-btn">Clear</button>
      </div>
    `;

    document.getElementById('send-btn').addEventListener('click', () => this.sendMessage());
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') this.sendMessage();
    });
    document.getElementById('clear-btn').addEventListener('click', () => this.clearSession());
  }

  async sendMessage() {
    const input = document.getElementById('chat-input');
    const prompt = input.value.trim();
    if (!prompt) return;

    // Add user message
    this.addMessage('user', prompt);
    input.value = '';

    try {
      const response = await fetch(`${this.apiUrl}/api/chat`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      const data = await response.json();

      // Add AI response
      this.addMessage('assistant', data.answer, data.sources);

    } catch (error) {
      console.error('Error:', error);
      this.addMessage('system', 'Error: Failed to send message');
    }
  }

  addMessage(role, content, sources = []) {
    this.messages.push({ role, content, sources });

    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `
      <div class="content">${content}</div>
      ${sources.length > 0 ? `
        <div class="sources">
          <strong>Sources:</strong>
          ${sources.map(src => `
            <div class="source">Page ${src.page}: ${src.content}</div>
          `).join('')}
        </div>
      ` : ''}
    `;

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  async clearSession() {
    try {
      await fetch(`${this.apiUrl}/api/clear-session`, {
        method: 'POST',
        credentials: 'include'
      });
      
      this.messages = [];
      document.getElementById('messages').innerHTML = '';
      alert('Conversation cleared!');
    } catch (error) {
      console.error('Error:', error);
    }
  }
}

// Usage:
// const chatbot = new Chatbot('http://localhost:8000', 'chatbot-container');


// ============================================
// Vue.js Example
// ============================================

/*
<template>
  <div class="chatbot">
    <div class="messages">
      <div v-for="(msg, idx) in messages" :key="idx" :class="['message', msg.role]">
        <div class="content">{{ msg.content }}</div>
        <div v-if="msg.sources && msg.sources.length > 0" class="sources">
          <strong>Sources:</strong>
          <div v-for="(src, i) in msg.sources" :key="i" class="source">
            Page {{ src.page }}: {{ src.content }}
          </div>
        </div>
      </div>
    </div>
    
    <div class="input-area">
      <input 
        v-model="input" 
        @keypress.enter="sendMessage"
        placeholder="Ask a question..."
        :disabled="loading"
      />
      <button @click="sendMessage" :disabled="loading">
        {{ loading ? 'Sending...' : 'Send' }}
      </button>
      <button @click="clearSession">Clear</button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      messages: [],
      input: '',
      loading: false,
      apiUrl: 'http://localhost:8000'
    };
  },
  methods: {
    async sendMessage() {
      if (!this.input.trim()) return;

      this.messages.push({ role: 'user', content: this.input });
      const prompt = this.input;
      this.input = '';
      this.loading = true;

      try {
        const response = await fetch(`${this.apiUrl}/api/chat`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt })
        });

        const data = await response.json();
        this.messages.push({ 
          role: 'assistant', 
          content: data.answer,
          sources: data.sources 
        });

      } catch (error) {
        console.error('Error:', error);
        alert('Failed to send message');
      } finally {
        this.loading = false;
      }
    },

    async clearSession() {
      try {
        await fetch(`${this.apiUrl}/api/clear-session`, {
          method: 'POST',
          credentials: 'include'
        });
        this.messages = [];
      } catch (error) {
        console.error('Error:', error);
      }
    }
  }
};
</script>
*/


// ============================================
// CSS Styling Example
// ============================================

const exampleCSS = `
.chatbot-container {
  max-width: 800px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  height: 600px;
  border: 1px solid #ddd;
  border-radius: 8px;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #f9f9f9;
}

.message {
  margin-bottom: 15px;
  padding: 12px;
  border-radius: 8px;
  max-width: 80%;
}

.message.user {
  background: #007bff;
  color: white;
  margin-left: auto;
}

.message.assistant {
  background: white;
  border: 1px solid #ddd;
}

.message .content {
  margin-bottom: 8px;
}

.sources {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid #eee;
  font-size: 0.9em;
  color: #666;
}

.source {
  margin: 5px 0;
  padding: 5px;
  background: #f5f5f5;
  border-radius: 4px;
}

.input-area {
  display: flex;
  padding: 15px;
  border-top: 1px solid #ddd;
  background: white;
}

.input-area input {
  flex: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-right: 10px;
}

.input-area button {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  background: #007bff;
  color: white;
  cursor: pointer;
  margin-left: 5px;
}

.input-area button:hover {
  background: #0056b3;
}

.input-area button:disabled {
  background: #ccc;
  cursor: not-allowed;
}
`;


// ============================================
// Advanced: Streaming Response (Optional)
// ============================================

async function sendMessageWithStreaming(prompt) {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt })
  });

  // Note: Current API doesn't support streaming yet
  // But here's how you'd implement it if we add SSE support:
  
  /*
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullResponse = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    fullResponse += chunk;
    
    // Update UI with partial response
    updateMessageInUI(fullResponse);
  }
  */

  return response.json();
}
