import base64
import copy
import json
import socket
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
from urllib import parse

import grpc
from grpc._cython import cygrpc

from pymilvus.decorators import ignore_unimplemented, retry_on_rpc_failure, upgrade_reminder
from pymilvus.exceptions import (
    AmbiguousIndexName,
    DescribeCollectionException,
    ErrorCode,
    ExceptionsMessage,
    MilvusException,
    ParamError,
)
from pymilvus.grpc_gen import common_pb2, milvus_pb2_grpc
from pymilvus.grpc_gen import milvus_pb2 as milvus_types
from pymilvus.settings import Config

from . import entity_helper, interceptor, ts_utils, utils
from .abstract import AnnSearchRequest, BaseRanker, CollectionSchema, MutationResult, SearchResult
from .asynch import (
    CreateIndexFuture,
    FlushFuture,
    LoadPartitionsFuture,
    MutationFuture,
    SearchFuture,
)
from .check import (
    check_pass_param,
    is_legal_host,
    is_legal_port,
)
from .constants import ITERATOR_SESSION_TS_FIELD
from .prepare import Prepare
from .types import (
    BulkInsertState,
    CompactionPlans,
    CompactionState,
    DatabaseInfo,
    DataType,
    ExtraList,
    GrantInfo,
    Group,
    IndexState,
    LoadState,
    Plan,
    PrivilegeGroupInfo,
    Replica,
    ResourceGroupConfig,
    ResourceGroupInfo,
    RoleInfo,
    Shard,
    State,
    Status,
    UserInfo,
    get_cost_extra,
)
from .utils import (
    check_invalid_binary_vector,
    check_status,
    get_server_type,
    is_successful,
    len_of,
)

exRequest = None

class AsyncGrpcHandler:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        uri: str = Config.GRPC_URI,
        host: str = "",
        port: str = "",
        channel: Optional[grpc.Channel] = None,
        **kwargs,
    ) -> None:
        # 11.28 EDITED
        self._async_stub = None
        self._async_channel = None

        addr = kwargs.get("address")
        self._address = addr if addr is not None else self.__get_address(uri, host, port)
        self._log_level = None
        self._request_id = None
        self._user = kwargs.get("user")
        self._setup_grpc_channel()
        self.callbacks = []


    def __get_address(self, uri: str, host: str, port: str) -> str:
        if host != "" and port != "" and is_legal_host(host) and is_legal_port(port):
            return f"{host}:{port}"

        try:
            parsed_uri = parse.urlparse(uri)
        except Exception as e:
            raise ParamError(message=f"Illegal uri: [{uri}], {e}") from e
        return parsed_uri.netloc

    def _set_authorization(self, **kwargs):
        secure = kwargs.get("secure", False)
        if not isinstance(secure, bool):
            raise ParamError(message="secure must be bool type")
        self._secure = secure
        self._client_pem_path = kwargs.get("client_pem_path", "")
        self._client_key_path = kwargs.get("client_key_path", "")
        self._ca_pem_path = kwargs.get("ca_pem_path", "")
        self._server_pem_path = kwargs.get("server_pem_path", "")
        self._server_name = kwargs.get("server_name", "")

        self._authorization_interceptor = None
        
    def __enter__(self):
        return self

    def __exit__(self: object, exc_type: object, exc_val: object, exc_tb: object):
        pass

    def close(self):
        pass

    # 11.28 EDITED 初始化 self._async_channel
    def _setup_grpc_channel(self):
        """Create a ddl grpc channel"""
        opts = [
                (cygrpc.ChannelArgKey.max_send_message_length, -1),
                (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                ("grpc.enable_retries", 1),
                ("grpc.keepalive_time_ms", 55000),
            ]

        self._async_channel = grpc.aio.insecure_channel(
                self._address,
                options=opts,
            )
        self._async_stub = milvus_pb2_grpc.MilvusServiceStub(self._async_channel)


    # 11.28 EDITED ADD
    async def _async_execute_search(
        self, request: milvus_types.SearchRequest, timeout: Optional[float] = None, **kwargs
    ):
        try:
            # 11.28 EDITED
            # response = self._stub.Search(request, timeout=timeout)
            response = await self._async_stub.Search(request, timeout=timeout)
            # 异步结果没有status
            # check_status(response.status)
            round_decimal = kwargs.get("round_decimal", -1)
            return SearchResult(
                response.results,
                round_decimal,
                status=response.status,
                session_ts=response.session_ts,
            )
        except Exception as e:
            raise e from e

    # EDITED ADD
    @retry_on_rpc_failure()
    async def async_search(
        self,
        collection_name: str,
        data: Union[List[List[float]], utils.SparseMatrixInputType],
        anns_field: str,
        param: Dict,
        limit: int,
        expression: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        global exRequest 
        if exRequest is None:
            exRequest = Prepare.search_requests_with_expr(
                collection_name,
                data,
                anns_field,
                param,
                limit,
                expression,
                partition_names,
                output_fields,
                round_decimal,
                **kwargs,
            )
        return await self._async_execute_search(exRequest, timeout, round_decimal=round_decimal, **kwargs)
