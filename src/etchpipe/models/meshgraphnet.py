"""C3: 모델 A — nvidia-physicsnemo MeshGraphNet 래퍼(W4-03).

`physicsnemo.models.meshgraphnet.meshgraphnet.MeshGraphNet.forward`는
`(node_features, edge_features, graph: torch_geometric.data.Data) -> (N, output_dim)`을
요구한다(`graph`는 `edge_index`를 담아 내부 메시지 전달(torch_scatter)에 쓰임 — 실측 확인
2026-11-09, `etch-train:0.1` 컨테이너에서 physicsnemo 2.2.2 소스로 직접 조회,
`reports/evidence/W4-03_meshgraphnet_overfit.md`).

`train/loop.py`의 `GraphDisplacementModel` 프로토콜(`model(node, edge_index, edge) ->
displacement`)에 맞춰 `graph` 구성을 감춘다. `nn.Module`을 상속해 `optimizer =
torch.optim.Adam(wrapper.parameters(), ...)`가 그대로 동작한다.

입출력 차원은 설계서 §3 C3: `input_dim_nodes=14`(`etl.to_graph` 출력과 일치),
`input_dim_edges=3`, `output_dim=2`(변위 dx,dy).

`processor_size=12`(기본값): GPU 팀원이 전달한 MGN 초안 체크포인트(`runs/mgn_draft/ckpt_mgn.pt`)의
`processor.processor_layers.0~11` 실측 레이어 수에 맞춤(2026-11-09, 설계서 §3 C3 제안 범위
8~15 안에 포함) — 이전 기본값 6(체크리스트 초안 W4-04 제안)과는 다름, 이 체크포인트를
`load_state_dict`로 바로 불러오려면 반드시 일치해야 함.
"""

import torch
from physicsnemo.models.meshgraphnet import MeshGraphNet
from torch import nn
from torch_geometric.data import Data


class MeshGraphNetModel(nn.Module):
    def __init__(
        self,
        input_dim_nodes: int = 14,
        input_dim_edges: int = 3,
        output_dim: int = 2,
        processor_size: int = 12,
        hidden_dim_processor: int = 64,
    ) -> None:
        super().__init__()
        self.net = MeshGraphNet(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            processor_size=processor_size,
            hidden_dim_processor=hidden_dim_processor,
        )

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        graph = Data(edge_index=edge_index, num_nodes=node.shape[0])
        return self.net(node, edge, graph)
