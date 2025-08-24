import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import path from 'path';

// gRPC connection details
const GRPC_HOST = process.env.SUPER_ALITA_GRPC_HOST || 'localhost';
const GRPC_PORT = process.env.SUPER_ALITA_GRPC_PORT || '50051';
const GRPC_ADDRESS = `${GRPC_HOST}:${GRPC_PORT}`;

// Optional TLS configuration
const USE_TLS = process.env.SUPER_ALITA_GRPC_USE_TLS === '1';
const credentials = USE_TLS
  ? grpc.credentials.createSsl()
  : grpc.credentials.createInsecure();

// Load protobuf definitions
const PROTO_PATH = path.resolve(
  __dirname,
  '../../../src/core/mangle/proto/super_alita.proto',
);

const packageDef = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});

const protoDescriptor = grpc.loadPackageDefinition(packageDef) as any;
const client = new protoDescriptor.super_alita.SuperAlitaAgent(
  GRPC_ADDRESS,
  credentials,
);

function promisify<TReq, TRes>(method: any, request: TReq): Promise<TRes> {
  return new Promise((resolve, reject) => {
    method.call(client, request, (err: grpc.ServiceError, res: TRes) => {
      if (err) {
        reject(err);
      } else {
        resolve(res);
      }
    });
  });
}

export const getHealth = (): Promise<any> =>
  promisify(client.GetHealth, {});

export const getStatus = (): Promise<any> =>
  promisify(client.GetStatus, {});

export const processTask = (req: any): Promise<any> =>
  promisify(client.ProcessTask, req);

export const kgQuery = (req: any): Promise<any> =>
  promisify(client.QueryKnowledgeGraph, req);

export const banditDecide = (req: any): Promise<any> =>
  promisify(client.MakeDecision, req);

export const banditFeedback = (req: any): Promise<any> =>
  promisify(client.ProvideFeedback, req);

