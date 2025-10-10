"""Backfill Mangle protobuf artifacts when generated stubs lack them."""

from __future__ import annotations

import logging
from typing import Any

import grpc
from google.protobuf import descriptor_pb2, descriptor_pool
from google.protobuf.internal import builder as _builder

from . import super_alita_pb2 as pb2
from . import super_alita_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


def _ensure_mangle_messages() -> None:
    pool = descriptor_pool.Default()
    try:
        pool.FindMessageTypeByName("super_alita.MangleProgramRequest")
        pool.FindMessageTypeByName("super_alita.MangleProgramResponse")
        return
    except KeyError:
        pass

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "super_alita_mangle_patch.proto"
    file_proto.package = "super_alita"
    file_proto.syntax = "proto3"

    req = file_proto.message_type.add()
    req.name = "MangleProgramRequest"
    field = req.field.add()
    field.name = "program"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field = req.field.add()
    field.name = "timeout_seconds"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    resp = file_proto.message_type.add()
    resp.name = "MangleProgramResponse"
    field = resp.field.add()
    field.name = "success"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BOOL
    field = resp.field.add()
    field.name = "error_message"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field = resp.field.add()
    field.name = "json_results"
    field.number = 3
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field = resp.field.add()
    field.name = "count"
    field.number = 4
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT32
    field = resp.field.add()
    field.name = "execution_time"
    field.number = 5
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    serialized = file_proto.SerializeToString()
    file_desc = pool.AddSerializedFile(serialized)
    logger.warning(
        "super_alita.proto artifacts missing Mangle definitions; applied runtime patch. Regenerate via protoc when possible."
    )
    _builder.BuildMessageAndEnumDescriptors(file_desc, pb2.__dict__)
    _builder.BuildTopDescriptorsAndMessages(
        file_desc, "super_alita_pb2", pb2.__dict__
    )


def _ensure_stub_bindings() -> None:
    if not hasattr(pb2, "MangleProgramRequest") or not hasattr(
        pb2, "MangleProgramResponse"
    ):
        return

    if (
        getattr(pb2_grpc.SuperAlitaAgentStub, "_mangle_query_patched", False)
        is False
    ):
        original_init = pb2_grpc.SuperAlitaAgentStub.__init__

        def _patched_init(self: Any, channel: grpc.Channel) -> None:
            original_init(self, channel)
            self.MangleQuery = channel.unary_unary(
                "/super_alita.SuperAlitaAgent/MangleQuery",
                request_serializer=pb2.MangleProgramRequest.SerializeToString,
                response_deserializer=pb2.MangleProgramResponse.FromString,
                _registered_method=True,
            )

        pb2_grpc.SuperAlitaAgentStub.__init__ = _patched_init  # type: ignore[assignment]
        pb2_grpc.SuperAlitaAgentStub._mangle_query_patched = True  # type: ignore[attr-defined]

    if not hasattr(pb2_grpc.SuperAlitaAgentServicer, "MangleQuery"):

        def _unimplemented(
            self: Any, request: Any, context: grpc.ServicerContext
        ) -> None:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Method not implemented!")
            raise NotImplementedError("Method not implemented!")

        pb2_grpc.SuperAlitaAgentServicer.MangleQuery = _unimplemented  # type: ignore[attr-defined]

    if (
        getattr(
            pb2_grpc.add_SuperAlitaAgentServicer_to_server,
            "_mangle_query_patched",
            False,
        )
        is False
    ):
        original_add = pb2_grpc.add_SuperAlitaAgentServicer_to_server

        def _patched_add(servicer: Any, server: grpc.Server) -> None:
            original_add(servicer, server)
            if not hasattr(servicer, "MangleQuery"):
                return
            handler = grpc.unary_unary_rpc_method_handler(
                servicer.MangleQuery,
                request_deserializer=pb2.MangleProgramRequest.FromString,
                response_serializer=pb2.MangleProgramResponse.SerializeToString,
            )
            generic_handler = grpc.method_handlers_generic_handler(
                "super_alita.SuperAlitaAgent",
                {"MangleQuery": handler},
            )
            server.add_generic_rpc_handlers((generic_handler,))

        pb2_grpc.add_SuperAlitaAgentServicer_to_server = _patched_add  # type: ignore[assignment]
        pb2_grpc.add_SuperAlitaAgentServicer_to_server._mangle_query_patched = True  # type: ignore[attr-defined]


_ensure_mangle_messages()
_ensure_stub_bindings()
