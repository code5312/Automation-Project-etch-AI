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

`processor_size=6`(기본값, [확인] 2026-11-09): physicsnemo 소스 실측(`meshgraphnet.py`의
`layers = list(chain(*zip(edge_blocks, node_blocks)))`) — `processor_size`개의 edge_block과
`processor_size`개의 node_block을 번갈아 쌓아 `processor_layers`는 **`2*processor_size`**개가
된다. GPU 팀원 체크포인트(`runs/mgn_draft/ckpt_mgn.pt`)의 `processor.processor_layers.0~11`
(12개)은 그래서 `processor_size=6`과 일치 — 한 번 `processor_size=12`로 잘못 바꿨다가
(인덱스 개수를 그대로 processor_size로 착각) `load_state_dict` 실패로 발견·정정함, 상세는
`reports/evidence/W4-06_rollout_vs_baseline.md` 참고.

`hidden_dim_node_encoder`/`hidden_dim_edge_encoder`/`hidden_dim_node_decoder`도
`hidden_dim_processor`와 같은 값으로 명시해 전달한다 — physicsnemo `MeshGraphNet`은 이
셋을 `hidden_dim_processor`와 **별도로** 받고 기본값이 각각 128이라, 전달하지 않으면
`hidden_dim_processor=64`를 줘도 인코더/디코더는 128로 만들어져 체크포인트(전부 64)
로드가 `load_state_dict` 크기 불일치로 실패한다(2026-11-09, W4-06에서 실제로 겪고 발견).
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
        processor_size: int = 6,
        hidden_dim_processor: int = 64,
    ) -> None:
        super().__init__()
        self.net = MeshGraphNet(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            processor_size=processor_size,
            hidden_dim_processor=hidden_dim_processor,
            hidden_dim_node_encoder=hidden_dim_processor,
            hidden_dim_edge_encoder=hidden_dim_processor,
            hidden_dim_node_decoder=hidden_dim_processor,
        )

    def forward(
        self, node: torch.Tensor, edge_index: torch.Tensor, edge: torch.Tensor
    ) -> torch.Tensor:
        graph = Data(edge_index=edge_index, num_nodes=node.shape[0])
        return self.net(node, edge, graph)
