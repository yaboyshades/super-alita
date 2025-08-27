import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import path from "path";

// gRPC client configuration for Super Alita
const GRPC_HOST = process.env.SUPER_ALITA_GRPC_HOST || "localhost";
const GRPC_PORT = process.env.SUPER_ALITA_GRPC_PORT || "50051";
const PROTO_PATH = path.resolve(
  __dirname,
  "../../../src/core/mangle/proto/super_alita.proto",
);

const pkgDef = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});
const proto = grpc.loadPackageDefinition(pkgDef) as any;
const client = new proto.super_alita.SuperAlitaAgent(
  `${GRPC_HOST}:${GRPC_PORT}`,
  grpc.credentials.createInsecure(),
);

// Type definitions for our gRPC methods
interface HealthResponse {
  status: number;
  message: string;
  timestamp?: { seconds: number; nanos: number };
  details?: Record<string, string>;
}

interface StatusResponse {
  version: string;
  uptime: { seconds: number; nanos: number };
  active_plugins: number;
  total_tasks_processed: number;
  total_events_emitted: number;
  system_info: Record<string, string>;
}

interface TaskRequest {
  task_id: string;
  content: string;
  session_id: string;
  user_id: string;
  workspace: string;
  metadata: Record<string, any>;
  timeout_seconds?: number;
}

interface TaskResponse {
  task_id: string;
  result: string;
  success: boolean;
  error_message?: string;
  execution_time?: number;
}

interface KGQueryRequest {
  query: string;
  limit: number;
  filters?: Record<string, string>;
}

interface KGQueryResponse {
  nodes: Array<Record<string, any>>;
  edges: Array<Record<string, any>>;
  total_results: number;
}

interface BanditDecisionRequest {
  policy_id: string;
}

interface BanditDecisionResponse {
  decision_id: string;
  algorithm: string;
  action: string;
  confidence: number;
  expected_reward: number;
}

interface BanditFeedbackRequest {
  decision_id: string;
  reward: number;
  source: string;
}

interface BanditFeedbackResponse {
  success: boolean;
  updated_policy: string;
  new_confidence: number;
}

// gRPC client implementations
export async function getHealth(): Promise<HealthResponse> {
  return new Promise((resolve, reject) => {
    client.GetHealth(
      {},
      (err: grpc.ServiceError | null, res: HealthResponse) => {
        if (err) {
          reject(err);
          return;
        }
        resolve(res);
      },
    );
  });
}

export async function getStatus(): Promise<StatusResponse> {
  return new Promise((resolve, reject) => {
    client.GetStatus(
      {},
      (err: grpc.ServiceError | null, res: StatusResponse) => {
        if (err) {
          reject(err);
          return;
        }
        resolve(res);
      },
    );
  });
}

export async function processTask(request: TaskRequest): Promise<TaskResponse> {
  return new Promise((resolve, reject) => {
    client.ProcessTask(
      request,
      (err: grpc.ServiceError | null, res: TaskResponse) => {
        if (err) {
          reject(err);
          return;
        }
        resolve(res);
      },
    );
  });
}

export async function kgQuery(
  request: KGQueryRequest,
): Promise<KGQueryResponse> {
  return new Promise((resolve, reject) => {
    client.QueryKnowledgeGraph(
      request,
      (err: grpc.ServiceError | null, res: KGQueryResponse) => {
        if (err) {
          reject(err);
          return;
        }
        resolve(res);
      },
    );
  });
}

export async function banditDecide(
  request: BanditDecisionRequest,
): Promise<BanditDecisionResponse> {
  try {
    // TODO: Replace with actual gRPC call when protobuf issues are resolved
    const algorithms = ["thompson_sampling", "ucb1", "epsilon_greedy"];
    const actions = ["explore", "exploit", "random"];

    const algorithm = algorithms[Math.floor(Math.random() * algorithms.length)];
    const action = actions[Math.floor(Math.random() * actions.length)];

    return {
      decision_id: `decision_${Date.now()}`,
      algorithm,
      action,
      confidence: Math.random() * 0.5 + 0.5, // 0.5-1.0
      expected_reward: Math.random() * 0.4 + 0.6, // 0.6-1.0
    };
  } catch (error) {
    throw new Error(`Bandit decision failed: ${error}`);
  }
}

export async function banditFeedback(
  request: BanditFeedbackRequest,
): Promise<BanditFeedbackResponse> {
  try {
    // TODO: Replace with actual gRPC call when protobuf issues are resolved
    return {
      success: true,
      updated_policy: "thompson_sampling_v2",
      new_confidence: Math.max(0.1, Math.min(1.0, request.reward)),
    };
  } catch (error) {
    throw new Error(`Bandit feedback failed: ${error}`);
  }
}
