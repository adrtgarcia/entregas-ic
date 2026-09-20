"""
desafio2_amanda_matheus.py — entrega do Desafio 2
Entrega oficial: este é o desafio2/desafio2_nomes.py do nosso repositório.
Ideia: em vez de decorar Xavier/He, CALIBRAR o ganho numericamente para a ativação escolhida.
Se z ~ N(0, 1) e W ~ N(0, s^2), a pré-ativação da próxima camada tem variância
    fan_in * s^2 * E[f(z)^2]
Para mantê-la em 1 camada após camada:  s^2 = 1 / (fan_in * E[f(z)^2]).
Troque `ativacao` por outra função e a inicialização se ajusta sozinha.
Rode:  python harness_desafio2.py desafio2/desafio2_amanda_matheus.py --rapido

Escolha: LeakyReLU (negative_slope=0.01). Como a ReLU, é positivamente homogênea
(f(a*z)=a*f(z)), então preserva a variância em qualquer profundidade — sobrevive às
48 camadas. A calibração numérica dá E[f(z)^2] ~ 0.5, logo s^2 ~ 2/fan_in — ou seja,
REPRODUZ o He (2015) sem decorá-lo. O leak de 0.01 evita "neurônio morto". Testamos
SELU+LeCun (preserva a variância, mas diverge com o SGD deste harness) e GELU (explode
a variância no fundo por ter média != 0); a família ReLU foi a escolha estável.

INTEGRANTES DO GRUPO:
- Amanda Duarte Garcia - 12221BCC031
- Matheus Fiod Saliba - 12221BCC024

Referências da API do PyTorch usadas aqui:
  torch.nn.functional.leaky_relu  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.leaky_relu.html
  torch.Tensor.normal_       https://docs.pytorch.org/docs/stable/generated/torch.Tensor.normal_.html
  torch.Tensor.zero_         https://docs.pytorch.org/docs/stable/generated/torch.Tensor.zero_.html
  torch.no_grad              https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
  torch.Generator            https://docs.pytorch.org/docs/stable/generated/torch.Generator.html
  torch.nn.init (Xavier, He, calculate_gain — para comparar com a conta feita à mão)
                             https://docs.pytorch.org/docs/stable/nn.init.html
  outras ativações: relu, leaky_relu, elu, selu, silu em https://docs.pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
"""
import math
import torch


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, 0.01)   # homogênea; 0.01 evita neurônio morto


# E[f(z)^2] com z ~ N(0,1), estimado uma vez por amostragem (Monte Carlo)
_g = torch.Generator().manual_seed(0)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_g)).pow(2).mean().item()


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    # s^2 = 1/(fan_in*E[f^2]); para (Leaky)ReLU isso equivale ao He (~2/fan_in)
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    W.normal_(0.0, desvio)
    b.zero_()
