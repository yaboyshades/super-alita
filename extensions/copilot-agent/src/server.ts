import express from "express";
import getRawBody from "raw-body";
import {
  verifyAndParseRequest,
  getUserMessage,
  getUserConfirmation,
  createAckEvent,
  createTextEvent,
  createDoneEvent,
  createConfirmationEvent,
  createErrorsEvent,
  createReferencesEvent,
} from "@copilot-extensions/preview-sdk";

import { initSSE, sendSSE, sendDefault } from "./sse.js";
import {
  getHealth, getStatus, processTask, kgQuery, banditDecide, banditFeedback,
} from "./grpcClient.js";

const app = express();

// IMPORTANT: we stream SSE back; no body parser so we can verify on the raw buffer
app.post("/copilot", async (req, res) => {
  // 0) Prepare SSE stream
  initSSE(res);

  // 1) Verify & parse (Preview SDK)
  const raw = await getRawBody(req);
  const signature = req.header("x-github-copilot-signature") ?? "";
  const keyId = req.header("x-github-copilot-key-id") ?? "";
  
  // For development/testing, skip verification if no token provided
  let isValidRequest = true;
  let payload: any;
  
  if (process.env.GITHUB_TOKEN) {
    const result = await verifyAndParseRequest(
      raw.toString(), signature, keyId, { token: process.env.GITHUB_TOKEN }
    );
    isValidRequest = result.isValidRequest;
    payload = result.payload;
  } else {
    // Skip verification for development
    try {
      payload = JSON.parse(raw.toString());
      console.log('[copilot-agent] Development mode: skipping signature verification');
    } catch (e) {
      isValidRequest = false;
    }
  }

  if (!isValidRequest) {
    // Option A: Preview SDK helper (formats SSE)
    res.write(createErrorsEvent([{
      type: "agent",
      code: "UNVERIFIED",
      message: "Request could not be verified",
      identifier: "verification",
    }]));
    return res.end(createDoneEvent());

    // Option B (manual SSE fallback):
    // sendSSE(res, "copilot_errors", [{ type: "agent", code: "UNVERIFIED", message: "Request could not be verified", identifier: "verification" }]);
    // return res.end(); // default SSE 'done' isn't specified; SDK handles done marker for you
  }

  // 2) Acknowledge start (Preview SDK emits an SSE chunk internally)
  res.write(createAckEvent());

  // 3) Pull message / confirmations
  const userMessage = getUserMessage(payload);
  const confirmation = getUserConfirmation(payload);
  if (confirmation) {
    // You received a response to a prior confirmation dialog
    res.write(createTextEvent(`Confirmation received: \`${confirmation.accepted ? 'accepted' : 'dismissed'}\``));
  }

  try {
    const text = userMessage.trim();

    // ——— Examples of explicit SSE event types (manual + SDK) ———

    // A) Send a confirmation dialog (maps to event: copilot_confirmation)
    if (/^confirm\s/i.test(text)) {
      // SDK way:
      res.write(createConfirmationEvent({
        id: "turn-off-flag",
        title: "Turn off feature flag?",
        message: "Are you sure you wish to turn off the `copilot` feature flag?",
        // metadata included in the user's follow-up
        metadata: { feature: "copilot" },
      }));
      return res.end(createDoneEvent());

      // Manual fallback:
      // sendSSE(res, "copilot_confirmation", {
      //   type: "action",
      //   title: "Turn off feature flag",
      //   message: "Are you sure you wish to turn off the `copilot` feature flag?",
      //   confirmation: { id: "turn-off-flag" }
      // });
      // return res.end();
    }

    // B) Emit references (event: copilot_references)
    if (/^refs\s/i.test(text)) {
      // SDK way:
      res.write(createTextEvent("Attaching references…"));
      res.write(createReferencesEvent([
        {
          id: "snippet",
          type: "blackbeard.story",
          data: { file: "story.go", start: "0", end: "13", content: "func main()..."},
          is_implicit: false,
          metadata: {
            display_name: "Lines 1-13 from story.go",
            display_icon: "book",
            display_url: "http://blackbeard.com/story/1"
          }
        }
      ]));
      // The SDK handles SSE formatting for references.
      // Manual fallback:
      // sendSSE(res, "copilot_references", [ /* same object as above */ ]);

      res.write(createTextEvent("References sent."));
      return res.end(createDoneEvent());
    }

    // C) Emit errors (event: copilot_errors)
    if (/^error\s/i.test(text)) {
      // SDK way:
      res.write(createErrorsEvent([{
        type: "function",
        code: "recentchanges",
        message: "The repository does not exist",
        identifier: "github/hello-world",
      }]));
      return res.end(createDoneEvent());

      // Manual fallback:
      // sendSSE(res, "copilot_errors", [{ type:"function", code:"recentchanges", message:"The repository does not exist", identifier:"github/hello-world" }]);
      // return res.end();
    }

    // ——— Super Alita specific commands ———

    // Health check
    if (/^health$/i.test(text)) {
      res.write(createTextEvent("🔍 Checking Super Alita health…"));
      const h = await getHealth();
      res.write(createTextEvent(`**Health Status**: ${h.message} (status=${h.status})\n\n⏰ Timestamp: ${h.timestamp}`));
    } 
    
    // System status
    else if (/^status$/i.test(text)) {
      res.write(createTextEvent("📊 Gathering system status…"));
      const s = await getStatus();
      res.write(createTextEvent("**System Status**:\n```json\n" + JSON.stringify(s, null, 2) + "\n```"));
    } 
    
    // Knowledge graph query
    else if (/^kg\s+/i.test(text)) {
      const query = text.slice(3);
      res.write(createTextEvent(`🧠 Querying knowledge graph: \`${query}\`…`));
      const r = await kgQuery({ query, limit: 50 });
      res.write(createTextEvent("**Knowledge Graph Results**:\n```json\n" + JSON.stringify(r, null, 2) + "\n```"));
    } 
    
    // Multi-armed bandit decision
    else if (/^decide\s+/i.test(text)) {
      const policy_id = text.split(/\s+/)[1] ?? "default";
      res.write(createTextEvent(`🎯 Making bandit decision for policy: \`${policy_id}\`…`));
      const r = await banditDecide({ policy_id });
      res.write(createTextEvent("**Decision Result**:\n```json\n" + JSON.stringify(r, null, 2) + "\n```"));
    } 
    
    // Bandit feedback
    else if (/^reward\s+/i.test(text)) {
      const [, decision_id, rewardStr] = text.split(/\s+/, 3);
      const reward = Number(rewardStr);
      if (!decision_id || Number.isNaN(reward)) {
        res.write(createTextEvent("❌ **Usage**: `reward <decision_id> <value>`\n\nExample: `reward decision_12345 0.8`"));
      } else {
        res.write(createTextEvent(`📈 Providing feedback for decision \`${decision_id}\` with reward \`${reward}\`…`));
        const r = await banditFeedback({ decision_id, reward, source: "copilot-agent" });
        res.write(createTextEvent("**Feedback Result**:\n```json\n" + JSON.stringify(r, null, 2) + "\n```"));
      }
    } 
    
    // Help command
    else if (/^help$/i.test(text)) {
      const helpText = `# 🤖 Super Alita Agent Commands

## Basic Commands
- \`health\` - Check system health
- \`status\` - Get detailed system status  
- \`help\` - Show this help

## Knowledge Graph
- \`kg <query>\` - Query knowledge graph
- Example: \`kg machine learning\`

## Multi-Armed Bandit
- \`decide <policy_id>\` - Make optimization decision
- \`reward <decision_id> <value>\` - Provide feedback
- Example: \`decide exploration\` then \`reward decision_12345 0.8\`

## Testing
- \`confirm test\` - Test confirmation dialog
- \`refs test\` - Test references attachment
- \`error test\` - Test error reporting

## General Task Processing
- Any other text will be processed by the Cortex cognitive cycle`;
      
      res.write(createTextEvent(helpText));
    }
    
    // Default: general task processing through Cortex
    else {
      const taskId = `cp_${Date.now()}`;
      res.write(createTextEvent(`🧠 Processing task \`${taskId}\` through Cortex…`));
      const r = await processTask({
        task_id: taskId,
        content: text,
        session_id: payload.copilot_thread_id ?? "",
        user_id: "copilot_user",
        workspace: "copilot_workspace",
        metadata: { source: "copilot-agent" }
      });
      res.write(createTextEvent("**Cortex Processing Result**:\n```json\n" + JSON.stringify(r, null, 2) + "\n```"));
    }

    return res.end(createDoneEvent());
  } catch (err: any) {
    // SDK way:
    res.write(createErrorsEvent([{
      type: "agent",
      code: "INTERNAL",
      message: err?.message ?? String(err),
      identifier: "exception",
    }]));
    return res.end(createDoneEvent());

    // Manual fallback:
    // sendSSE(res, "copilot_errors", [{ type:"agent", code:"INTERNAL", message:String(err), identifier:"exception"}]);
    // return res.end();
  }
});

// Liveness
app.get("/healthz", (_req, res) => res.status(200).send("ok"));

const PORT = Number(process.env.PORT || 8787);
app.listen(PORT, '0.0.0.0', () => {
  console.log(`[copilot-agent] SSE server on :${PORT}`);
  console.log(`[copilot-agent] Health check: http://localhost:${PORT}/healthz`);
  console.log(`[copilot-agent] Copilot endpoint: http://localhost:${PORT}/copilot`);
});