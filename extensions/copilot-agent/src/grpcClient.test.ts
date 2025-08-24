import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import path from 'path';

const PROTO_PATH = path.resolve(
  __dirname,
  '../../../src/core/mangle/proto/super_alita.proto',
);

describe('gRPC client integration', () => {
  let server: grpc.Server;
  let client: typeof import('./grpcClient');

  beforeAll((done) => {
    const pkgDef = protoLoader.loadSync(PROTO_PATH, {
      keepCase: true,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true,
    });
    const proto = grpc.loadPackageDefinition(pkgDef) as any;
    server = new grpc.Server();
    server.addService(proto.super_alita.SuperAlitaAgent.service, {
      GetHealth: (_: unknown, cb: any) =>
        cb(null, { status: 0, message: 'ok', timestamp: {} }),
      ProcessTask: (call: any, cb: any) =>
        cb(null, {
          task_id: call.request.task_id,
          result: 'done',
          success: true,
        }),
    });
    server.bindAsync(
      '0.0.0.0:0',
      grpc.ServerCredentials.createInsecure(),
      (err, port) => {
        if (err) return done(err);
        server.start();
        process.env.SUPER_ALITA_GRPC_HOST = 'localhost';
        process.env.SUPER_ALITA_GRPC_PORT = String(port);
        client = require('./grpcClient');
        done();
      },
    );
  });

  afterAll(() => {
    server.forceShutdown();
  });

  test('getHealth performs a round trip', async () => {
    const res = await client.getHealth();
    expect(res).toHaveProperty('message', 'ok');
  });

  test('processTask performs a round trip', async () => {
    const res = await client.processTask({
      task_id: 't1',
      content: 'do',
      session_id: 's1',
      user_id: 'u1',
      workspace: 'w1',
      metadata: {},
    });
    expect(res).toHaveProperty('task_id', 't1');
    expect(res).toHaveProperty('success', true);
  });
});

