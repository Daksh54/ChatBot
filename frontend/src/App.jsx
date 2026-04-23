// Author: Daksh Sharma 26434

import { useEffect, useRef, useState } from "react";
import axios from "axios";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  ArrowUp,
  Bot,
  DatabaseZap,
  FileText,
  FileUp,
  FolderKanban,
  History,
  House,
  LoaderCircle,
  MessageSquareText,
  Plus,
  Sparkles,
} from "lucide-react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const navItems = [
  { id: "home", label: "Home", icon: House },
  { id: "chat", label: "Chat", icon: MessageSquareText },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "workspaces", label: "Workspaces", icon: FolderKanban },
  { id: "history", label: "History", icon: History },
];

const promptIdeas = [
  "Summarize the current document.",
  "Compare the main points across sources.",
  "List the strongest cited evidence.",
];

const createMessageId = () =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random()}`;

function CopyCodeButton({ code }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200 transition hover:bg-white/10"
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function MarkdownMessage({ children }) {
  return (
    <Markdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ inline, className, children: codeChildren, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          const code = String(codeChildren).replace(/\n$/, "");

          if (inline) {
            return (
              <code className="rounded bg-white/10 px-1.5 py-0.5 text-sm" {...props}>
                {codeChildren}
              </code>
            );
          }

          return (
            <div className="my-4 overflow-hidden rounded-2xl border border-white/10">
              <div className="flex items-center justify-between bg-slate-950 px-4 py-2 text-xs uppercase tracking-[0.2em] text-slate-400">
                <span>{match?.[1] || "code"}</span>
                <CopyCodeButton code={code} />
              </div>
              <SyntaxHighlighter
                {...props}
                language={match?.[1]}
                style={oneDark}
                customStyle={{ margin: 0, borderRadius: 0, background: "#020617" }}
              >
                {code}
              </SyntaxHighlighter>
            </div>
          );
        },
        table({ children }) {
          return (
            <div className="my-4 overflow-x-auto">
              <table className="min-w-full text-sm">{children}</table>
            </div>
          );
        },
        thead({ children }) {
          return <thead className="bg-white/5">{children}</thead>;
        },
        th({ children }) {
          return <th className="border border-white/10 px-3 py-2 text-left font-medium">{children}</th>;
        },
        td({ children }) {
          return <td className="border border-white/10 px-3 py-2 align-top">{children}</td>;
        },
      }}
    >
      {children}
    </Markdown>
  );
}

function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
      <div>
        <h1 className="font-display text-3xl font-semibold text-white">{title}</h1>
        {subtitle ? <p className="mt-2 text-sm text-slate-400">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap gap-3">{actions}</div> : null}
    </div>
  );
}

function Panel({ title, subtitle, actions, children, className = "" }) {
  return (
    <section className={`rounded-[1.75rem] border border-white/10 bg-slate-950/50 p-5 backdrop-blur-xl ${className}`}>
      {(title || subtitle || actions) && (
        <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            {title ? <h2 className="text-lg font-semibold text-white">{title}</h2> : null}
            {subtitle ? <p className="mt-1 text-sm text-slate-400">{subtitle}</p> : null}
          </div>
          {actions ? <div className="flex flex-wrap gap-3">{actions}</div> : null}
        </div>
      )}
      {children}
    </section>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="rounded-[1.5rem] border border-white/10 bg-white/5 p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}

function EmptyState({ title, description }) {
  return (
    <div className="rounded-[1.5rem] border border-dashed border-white/10 bg-white/5 px-5 py-10 text-center">
      <p className="text-base font-medium text-white">{title}</p>
      <p className="mt-2 text-sm text-slate-400">{description}</p>
    </div>
  );
}

function DocumentPreview({ selectedDocument, viewerPage }) {
  if (!selectedDocument) {
    return <EmptyState title="No document selected" description="Choose a document from the list." />;
  }

  if (selectedDocument.mime_type?.includes("pdf")) {
    return (
      <div className="overflow-hidden rounded-[1.5rem] border border-white/10 bg-slate-900/80">
        <iframe
          title="Document viewer"
          src={`${API_BASE_URL}/documents/${selectedDocument.id}/content#page=${viewerPage}`}
          className="h-[620px] w-full bg-white"
        />
      </div>
    );
  }

  return (
    <div className="rounded-[1.5rem] border border-white/10 bg-white/5 p-6">
      <p className="text-sm text-slate-400">{selectedDocument.mime_type || "Document"}</p>
      <h3 className="mt-2 text-2xl font-semibold text-white">{selectedDocument.name}</h3>
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <StatCard label="Pages" value={selectedDocument.page_count} />
        <StatCard label="Chunks" value={selectedDocument.chunk_count} />
      </div>
    </div>
  );
}

function HomePage({
  workspaces,
  documents,
  messages,
  selectedWorkspace,
  selectedDocument,
  activeTask,
  statusText,
  setActivePage,
}) {
  const lastAssistantMessage = [...messages].reverse().find((message) => message.role === "assistant");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Home"
        subtitle="Quick overview"
        actions={
          <>
            <button
              type="button"
              onClick={() => setActivePage("chat")}
              className="rounded-full bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300"
            >
              Open chat
            </button>
            <button
              type="button"
              onClick={() => setActivePage("documents")}
              className="rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm text-slate-100 transition hover:bg-white/10"
            >
              Open documents
            </button>
          </>
        }
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Workspaces" value={workspaces.length} />
        <StatCard label="Documents" value={documents.length} />
        <StatCard label="Messages" value={messages.length} />
        <StatCard label="Status" value={activeTask ? activeTask.status : "Ready"} />
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_360px]">
        <Panel title="Current selection">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
              <p className="text-sm text-slate-400">Workspace</p>
              <p className="mt-2 text-lg font-semibold text-white">
                {selectedWorkspace?.name || "No workspace"}
              </p>
            </div>
            <div className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
              <p className="text-sm text-slate-400">Document</p>
              <p className="mt-2 text-lg font-semibold text-white">
                {selectedDocument?.name || "No document"}
              </p>
            </div>
            <div className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4 md:col-span-2">
              <p className="text-sm text-slate-400">Status</p>
              <p className="mt-2 text-white">{statusText}</p>
            </div>
          </div>
        </Panel>

        <Panel title="Quick actions">
          <div className="grid gap-3">
            {navItems
              .filter((item) => item.id !== "home")
              .map((item) => {
                const Icon = item.icon;

                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setActivePage(item.id)}
                    className="flex items-center gap-3 rounded-[1.25rem] border border-white/10 bg-white/5 px-4 py-4 text-left text-sm text-slate-200 transition hover:bg-white/10"
                  >
                    <Icon className="h-4 w-4 text-cyan-300" />
                    {item.label}
                  </button>
                );
              })}
          </div>
        </Panel>
      </div>

      <Panel title="Latest reply">
        {lastAssistantMessage ? (
          <div className="text-sm leading-7 text-slate-200">
            <MarkdownMessage>{lastAssistantMessage.content}</MarkdownMessage>
          </div>
        ) : (
          <EmptyState title="No replies yet" description="Start a chat to see recent answers here." />
        )}
      </Panel>
    </div>
  );
}

function WorkspacesPage({
  workspaceName,
  setWorkspaceName,
  handleCreateWorkspace,
  isCreatingWorkspace,
  workspaces,
  selectedWorkspaceId,
  setSelectedWorkspaceId,
}) {
  return (
    <div className="space-y-6">
      <PageHeader title="Workspaces" subtitle="Create and switch workspaces" />

      <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
        <Panel title="New workspace">
          <div className="space-y-3">
            <input
              value={workspaceName}
              onChange={(event) => setWorkspaceName(event.target.value)}
              placeholder="Workspace name"
              className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/50 focus:bg-white/10"
            />
            <button
              type="button"
              onClick={() => void handleCreateWorkspace()}
              disabled={isCreatingWorkspace}
              className="inline-flex items-center gap-2 rounded-full bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <Plus className="h-4 w-4" />
              Create
            </button>
          </div>
        </Panel>

        <Panel title="Workspace list">
          <div className="grid gap-3">
            {workspaces.length ? (
              workspaces.map((workspace) => {
                const isActive = workspace.id === selectedWorkspaceId;

                return (
                  <button
                    key={workspace.id}
                    type="button"
                    onClick={() => setSelectedWorkspaceId(workspace.id)}
                    className={`rounded-[1.25rem] border px-4 py-4 text-left transition ${
                      isActive
                        ? "border-cyan-300/40 bg-cyan-300/12 text-white"
                        : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                    }`}
                  >
                    <p className="font-medium">{workspace.name}</p>
                    <p className="mt-1 text-sm text-slate-400">
                      {workspace.description || "No description"}
                    </p>
                  </button>
                );
              })
            ) : (
              <EmptyState title="No workspaces" description="Create one to get started." />
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function DocumentsPage({
  selectedWorkspace,
  documents,
  selectedDocumentId,
  setSelectedDocumentId,
  viewerPage,
  setViewerPage,
  selectedDocument,
  activeTask,
  isUploading,
  handleUpload,
  fileInputRef,
}) {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Documents"
        subtitle={selectedWorkspace ? selectedWorkspace.name : "Select a workspace first"}
        actions={
          <label className="inline-flex cursor-pointer items-center gap-2 rounded-full bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300">
            {isUploading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <FileUp className="h-4 w-4" />}
            Upload
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.csv,.xlsx,.xls"
              onChange={(event) => void handleUpload(event)}
              className="hidden"
            />
          </label>
        }
      />

      <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
        <Panel title="Files">
          <div className="space-y-3">
            {activeTask ? (
              <div className="rounded-[1.25rem] border border-cyan-300/20 bg-cyan-400/10 px-4 py-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-medium text-white">{activeTask.status}</p>
                  <p className="text-xs uppercase tracking-[0.2em] text-cyan-100/80">
                    {activeTask.progress}%
                  </p>
                </div>
                <p className="mt-2 text-sm text-cyan-50/80">{activeTask.phase}</p>
              </div>
            ) : null}

            {documents.length ? (
              documents.map((document) => {
                const isSelected = document.id === selectedDocumentId;

                return (
                  <button
                    key={document.id}
                    type="button"
                    onClick={() => {
                      setSelectedDocumentId(document.id);
                      setViewerPage(1);
                    }}
                    className={`w-full rounded-[1.25rem] border px-4 py-4 text-left transition ${
                      isSelected
                        ? "border-sky-400/50 bg-sky-400/12 text-white"
                        : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <p className="truncate font-medium">{document.name}</p>
                      <span className="rounded-full bg-white/10 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-300">
                        {document.status || "READY"}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-slate-400">
                      {document.page_count} pages • {document.chunk_count} chunks
                    </p>
                  </button>
                );
              })
            ) : (
              <EmptyState title="No documents" description="Upload a file in the selected workspace." />
            )}
          </div>
        </Panel>

        <Panel
          title={selectedDocument ? selectedDocument.name : "Preview"}
          subtitle={selectedDocument ? `Page ${viewerPage}` : ""}
        >
          <DocumentPreview selectedDocument={selectedDocument} viewerPage={viewerPage} />
        </Panel>
      </div>
    </div>
  );
}

function ChatPage({
  messages,
  selectedWorkspace,
  selectedDocument,
  messageInput,
  setMessageInput,
  handleSendMessage,
  handleMessageKeyDown,
  isStreaming,
  jumpToCitation,
  setActivePage,
  chatEndRef,
}) {
  return (
    <div className="space-y-6">
      <PageHeader title="Chat" subtitle={selectedWorkspace ? selectedWorkspace.name : "Select a workspace first"} />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Panel title="Conversation" className="flex min-h-[760px] flex-col">
          <div className="flex-1 space-y-4 overflow-y-auto pr-2">
            {messages.length ? (
              messages.map((message) => (
                <article
                  key={message.id}
                  className={`rounded-[1.5rem] border p-5 ${
                    message.role === "user"
                      ? "ml-10 border-cyan-400/20 bg-cyan-400/12 text-white"
                      : "mr-4 border-white/10 bg-white/5 text-slate-100"
                  }`}
                >
                  <div className="mb-4 flex items-center gap-3">
                    <div
                      className={`grid h-10 w-10 place-items-center rounded-2xl ${
                        message.role === "user" ? "bg-cyan-400 text-slate-950" : "bg-white/10 text-white"
                      }`}
                    >
                      {message.role === "user" ? <ArrowUp className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                    </div>
                    <div>
                      <p className="font-medium">{message.role === "user" ? "You" : "Assistant"}</p>
                      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                        {message.streaming ? "Streaming" : "Ready"}
                      </p>
                    </div>
                  </div>

                  {message.role === "assistant" ? (
                    <div className="text-sm leading-7 text-slate-200">
                      <MarkdownMessage>{message.content || "Thinking..."}</MarkdownMessage>
                    </div>
                  ) : (
                    <p className="whitespace-pre-wrap text-slate-50">{message.content}</p>
                  )}

                  {!!message.citations?.length && (
                    <div className="mt-5 flex flex-wrap gap-2">
                      {message.citations.map((citation, index) => (
                        <button
                          key={`${message.id}-${index}`}
                          type="button"
                          onClick={() => {
                            jumpToCitation(citation);
                            setActivePage("documents");
                          }}
                          className="rounded-full border border-sky-300/30 bg-sky-400/10 px-3 py-1.5 text-xs font-medium text-sky-100 transition hover:bg-sky-400/20"
                        >
                          Source {index + 1} • Page {citation.page_start}
                        </button>
                      ))}
                    </div>
                  )}
                </article>
              ))
            ) : (
              <EmptyState title="No messages" description="Send a message to start chatting." />
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="mt-5 rounded-[1.5rem] border border-white/10 bg-slate-900/80 p-3">
            <div className="flex gap-3">
              <textarea
                value={messageInput}
                onChange={(event) => setMessageInput(event.target.value)}
                onKeyDown={handleMessageKeyDown}
                placeholder="Ask a question..."
                className="min-h-[92px] flex-1 resize-none rounded-[1.25rem] border border-white/10 bg-white/5 px-4 py-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/40 focus:bg-white/10"
              />
              <button
                type="button"
                onClick={() => void handleSendMessage()}
                disabled={!messageInput.trim() || isStreaming}
                className="inline-flex w-16 items-center justify-center rounded-[1.25rem] bg-cyan-400 text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isStreaming ? <LoaderCircle className="h-5 w-5 animate-spin" /> : <ArrowUp className="h-5 w-5" />}
              </button>
            </div>
          </div>
        </Panel>

        <div className="space-y-6">
          <Panel title="Context">
            <div className="space-y-4">
              <div className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
                <p className="text-sm text-slate-400">Workspace</p>
                <p className="mt-2 text-base font-semibold text-white">
                  {selectedWorkspace?.name || "None"}
                </p>
              </div>
              <div className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
                <p className="text-sm text-slate-400">Document</p>
                <p className="mt-2 text-base font-semibold text-white">
                  {selectedDocument?.name || "None"}
                </p>
              </div>
            </div>
          </Panel>

          <Panel title="Prompts">
            <div className="space-y-3">
              {promptIdeas.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => setMessageInput(prompt)}
                  className="w-full rounded-[1.25rem] border border-white/10 bg-white/5 px-4 py-3 text-left text-sm text-slate-200 transition hover:bg-white/10"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function HistoryPage({ messages, jumpToCitation, setActivePage }) {
  return (
    <div className="space-y-6">
      <PageHeader title="History" subtitle="Recent messages in the current workspace" />

      <Panel title="Messages">
        {messages.length ? (
          <div className="space-y-4">
            {messages.map((message) => (
              <div key={message.id} className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="font-medium text-white">{message.role === "user" ? "You" : "Assistant"}</p>
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                    {message.citations?.length || 0} citations
                  </p>
                </div>
                <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-300">{message.content}</p>
                {!!message.citations?.length && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {message.citations.map((citation, index) => (
                      <button
                        key={`${message.id}-history-${index}`}
                        type="button"
                        onClick={() => {
                          jumpToCitation(citation);
                          setActivePage("documents");
                        }}
                        className="rounded-full border border-sky-300/30 bg-sky-400/10 px-3 py-1.5 text-xs font-medium text-sky-100 transition hover:bg-sky-400/20"
                      >
                        Source {index + 1} • Page {citation.page_start}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <EmptyState title="No history" description="Messages will appear here after you chat." />
        )}
      </Panel>
    </div>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState("home");
  const [workspaceName, setWorkspaceName] = useState("");
  const [messageInput, setMessageInput] = useState("");
  const [workspaces, setWorkspaces] = useState([]);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState("");
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [messages, setMessages] = useState([]);
  const [statusText, setStatusText] = useState("Ready");
  const [isCreatingWorkspace, setIsCreatingWorkspace] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeTask, setActiveTask] = useState(null);
  const [viewerPage, setViewerPage] = useState(1);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const bootstrappedRef = useRef(false);

  const selectedWorkspace = workspaces.find((workspace) => workspace.id === selectedWorkspaceId) || null;
  const selectedDocument = documents.find((document) => document.id === selectedDocumentId) || null;

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => {
    if (bootstrappedRef.current) {
      return;
    }

    bootstrappedRef.current = true;
    void bootstrap();
  }, []);

  useEffect(() => {
    if (!selectedWorkspaceId) {
      setDocuments([]);
      setMessages([]);
      return;
    }

    void (async () => {
      try {
        await Promise.all([
          loadDocuments(selectedWorkspaceId),
          loadMessages(selectedWorkspaceId),
        ]);
      } catch (error) {
        setStatusText(`Workspace load failed: ${error.message}`);
      }
    })();
  }, [selectedWorkspaceId]);

  useEffect(() => {
    if (!activeTask?.id || activeTask.status === "SUCCEEDED" || activeTask.status === "FAILED") {
      return;
    }

    const intervalId = window.setInterval(() => {
      void pollTaskStatus(activeTask.id);
    }, 2000);

    void pollTaskStatus(activeTask.id);

    return () => window.clearInterval(intervalId);
  }, [activeTask?.id, activeTask?.status, selectedWorkspaceId]);

  async function bootstrap() {
    try {
      const { data } = await axios.get(`${API_BASE_URL}/workspaces`);
      const items = data.items || [];
      setWorkspaces(items);

      if (items[0]) {
        setSelectedWorkspaceId(items[0].id);
      }

      if (!items.length) {
        const created = await createWorkspace("Main Workspace", "Default workspace");
        setSelectedWorkspaceId(created.id);
      }
    } catch (error) {
      setStatusText(`Unable to load workspace list: ${error.message}`);
    }
  }

  async function createWorkspace(name, description) {
    const { data } = await axios.post(`${API_BASE_URL}/workspaces`, {
      name,
      description,
    });

    setWorkspaces((prev) => {
      const exists = prev.some((workspace) => workspace.id === data.id);
      if (exists) {
        return prev.map((workspace) => (workspace.id === data.id ? data : workspace));
      }
      return [data, ...prev];
    });

    return data;
  }

  async function loadDocuments(workspaceId) {
    const { data } = await axios.get(`${API_BASE_URL}/workspaces/${workspaceId}/documents`);
    const items = data.items || [];

    setDocuments(items);
    setSelectedDocumentId((current) => {
      if (current && items.some((document) => document.id === current)) {
        return current;
      }
      return items[0]?.id || "";
    });
  }

  async function loadMessages(workspaceId) {
    const { data } = await axios.get(`${API_BASE_URL}/workspaces/${workspaceId}/messages`);
    const history = [];

    for (const item of data.items || []) {
      history.push({
        id: `${item.id}-user`,
        role: "user",
        content: item.user_message,
        citations: [],
      });
      history.push({
        id: `${item.id}-assistant`,
        role: "assistant",
        content: item.assistant_message,
        citations: item.citations || [],
      });
    }

    setMessages(history);
  }

  function upsertDocument(document) {
    if (!document) {
      return;
    }

    setDocuments((prev) => {
      const exists = prev.some((item) => item.id === document.id);
      if (exists) {
        return prev.map((item) => (item.id === document.id ? document : item));
      }
      return [document, ...prev];
    });
  }

  async function pollTaskStatus(taskId) {
    try {
      const { data } = await axios.get(`${API_BASE_URL}/tasks/${taskId}`);
      const nextTask = data.task;
      setActiveTask(nextTask);
      upsertDocument(data.document);

      if (nextTask.status === "PROCESSING") {
        setStatusText(`${nextTask.phase} (${nextTask.progress}%)`);
      }

      if (nextTask.status === "SUCCEEDED") {
        if (selectedWorkspaceId) {
          await loadDocuments(selectedWorkspaceId);
        }

        if (data.document?.id) {
          setSelectedDocumentId(data.document.id);
        }

        setViewerPage(1);
        setStatusText(`${data.document?.name || "Document"} indexed.`);
      }

      if (nextTask.status === "FAILED") {
        setStatusText(`Indexing failed: ${nextTask.error || "Unknown worker error"}`);
      }
    } catch (error) {
      setStatusText(`Task polling failed: ${error.message}`);
    }
  }

  async function handleCreateWorkspace() {
    if (!workspaceName.trim() || isCreatingWorkspace) {
      return;
    }

    setIsCreatingWorkspace(true);

    try {
      const workspace = await createWorkspace(workspaceName.trim(), "Workspace");
      setWorkspaceName("");
      setSelectedWorkspaceId(workspace.id);
      setStatusText(`Workspace "${workspace.name}" created.`);
    } catch (error) {
      setStatusText(`Workspace creation failed: ${error.message}`);
    } finally {
      setIsCreatingWorkspace(false);
    }
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file || !selectedWorkspaceId) {
      return;
    }

    setIsUploading(true);
    setStatusText(`Uploading ${file.name}...`);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("workspace_id", selectedWorkspaceId);

      const { data } = await axios.post(`${API_BASE_URL}/documents/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setActiveTask(data.task);
      upsertDocument(data.document);
      setSelectedDocumentId(data.document.id);
      setViewerPage(1);
      setStatusText(`${data.document.name} uploaded.`);
    } catch (error) {
      setStatusText(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  async function handleSendMessage() {
    const trimmed = messageInput.trim();
    if (!trimmed || !selectedWorkspaceId || isStreaming) {
      return;
    }

    const userMessage = {
      id: createMessageId(),
      role: "user",
      content: trimmed,
      citations: [],
    };
    const assistantMessageId = createMessageId();

    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: assistantMessageId, role: "assistant", content: "", citations: [], streaming: true },
    ]);
    setMessageInput("");
    setIsStreaming(true);
    setStatusText("Generating answer...");

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspace_id: selectedWorkspaceId,
          document_id: selectedDocumentId || null,
          message: trimmed,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const eventChunk of events) {
          const line = eventChunk.split("\n").find((entry) => entry.startsWith("data: "));

          if (!line) {
            continue;
          }

          const payload = JSON.parse(line.slice(6));

          if (payload.type === "context") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantMessageId
                  ? { ...message, citations: payload.citations || [] }
                  : message
              )
            );
          }

          if (payload.type === "assistant_chunk") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantMessageId
                  ? { ...message, content: `${message.content}${payload.content}` }
                  : message
              )
            );
          }

          if (payload.type === "error") {
            throw new Error(payload.message);
          }

          if (payload.type === "assistant_done") {
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantMessageId ? { ...message, streaming: false } : message
              )
            );
            setStatusText("Answer complete.");
          }
        }
      }
    } catch (error) {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                content: `Streaming error: ${error.message}`,
                streaming: false,
              }
            : message
        )
      );
      setStatusText(`Streaming failed: ${error.message}`);
    } finally {
      setIsStreaming(false);
    }
  }

  function handleMessageKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSendMessage();
    }
  }

  function jumpToCitation(citation) {
    setViewerPage(citation.page_start || 1);
    if (citation.document_id) {
      setSelectedDocumentId(citation.document_id);
    }
  }

  function renderPage() {
    switch (activePage) {
      case "chat":
        return (
          <ChatPage
            messages={messages}
            selectedWorkspace={selectedWorkspace}
            selectedDocument={selectedDocument}
            messageInput={messageInput}
            setMessageInput={setMessageInput}
            handleSendMessage={handleSendMessage}
            handleMessageKeyDown={handleMessageKeyDown}
            isStreaming={isStreaming}
            jumpToCitation={jumpToCitation}
            setActivePage={setActivePage}
            chatEndRef={chatEndRef}
          />
        );
      case "documents":
        return (
          <DocumentsPage
            selectedWorkspace={selectedWorkspace}
            documents={documents}
            selectedDocumentId={selectedDocumentId}
            setSelectedDocumentId={setSelectedDocumentId}
            viewerPage={viewerPage}
            setViewerPage={setViewerPage}
            selectedDocument={selectedDocument}
            activeTask={activeTask}
            isUploading={isUploading}
            handleUpload={handleUpload}
            fileInputRef={fileInputRef}
          />
        );
      case "workspaces":
        return (
          <WorkspacesPage
            workspaceName={workspaceName}
            setWorkspaceName={setWorkspaceName}
            handleCreateWorkspace={handleCreateWorkspace}
            isCreatingWorkspace={isCreatingWorkspace}
            workspaces={workspaces}
            selectedWorkspaceId={selectedWorkspaceId}
            setSelectedWorkspaceId={setSelectedWorkspaceId}
          />
        );
      case "history":
        return (
          <HistoryPage
            messages={messages}
            jumpToCitation={jumpToCitation}
            setActivePage={setActivePage}
          />
        );
      default:
        return (
          <HomePage
            workspaces={workspaces}
            documents={documents}
            messages={messages}
            selectedWorkspace={selectedWorkspace}
            selectedDocument={selectedDocument}
            activeTask={activeTask}
            statusText={statusText}
            setActivePage={setActivePage}
          />
        );
    }
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.16),_transparent_32%),linear-gradient(135deg,_#020617,_#0f172a_46%,_#111827)] text-slate-100">
      <div className="mx-auto max-w-[1440px] px-4 py-4 lg:px-6">
        <header className="sticky top-4 z-20 mb-6 rounded-[1.75rem] border border-white/10 bg-slate-950/70 px-5 py-4 backdrop-blur-xl">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex items-center gap-3">
              <div className="grid h-11 w-11 place-items-center rounded-2xl bg-cyan-400/15 text-cyan-300">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <p className="font-display text-xl font-semibold text-white">ChatApp</p>
                <p className="text-sm text-slate-400">
                  {selectedWorkspace?.name || "No workspace selected"}
                </p>
              </div>
            </div>

            <nav className="flex flex-wrap gap-2">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = activePage === item.id;

                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setActivePage(item.id)}
                    className={`inline-flex items-center gap-2 rounded-full px-4 py-2.5 text-sm transition ${
                      isActive
                        ? "bg-cyan-400 text-slate-950"
                        : "border border-white/10 bg-white/5 text-slate-200 hover:bg-white/10"
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </button>
                );
              })}
            </nav>

            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300">
              <DatabaseZap className="h-4 w-4 text-cyan-300" />
              {statusText}
            </div>
          </div>
        </header>

        {renderPage()}
      </div>
    </div>
  );
}
