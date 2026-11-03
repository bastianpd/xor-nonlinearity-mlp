# Porque é que a não linearidade é necessária: resolver o XOR com um MLP ReLU em NumPy

**Um perceptrão não consegue aprender o XOR. Uma rede ReLU 2→4→1 consegue.** Este repositório mostra porquê: demonstração, derivação completa de backpropagation, experiências com regularização L1 e simulações interativas.

- Apresentação interativa: https://SEU-USUARIO.github.io/xor-nonlinearity-mlp/
- Notebook (corre na tua própria sessão Colab): [abrir no Colab](https://colab.research.google.com/github/bastianpd/xor-nonlinearity-mlp/blob/main/notebooks/xor_WhyNonLinearity_MLP.ipynb)
- Página para o portfólio: [`portfolio/xor-nonlinearity.html`](portfolio/xor-nonlinearity.html)

Autor: **Pedro de Sousa Sebastião** · CPGEI, UTFPR.

## Resultados (seed = 67, 5000 épocas, lr = 0,5)

| Modelo | Acc. treino | Acc. teste | BCE teste | Fração A₁ = 0 |
|---|---|---|---|---|
| Regressão logística | 0,657 | **0,540** | 0,7067 | n/a |
| MLP, λ = 0 | 1,000 | **0,980** | 0,0332 | 0,538 |
| MLP, λ = 0,005 | 1,000 | **0,980** | 0,0565 | 0,620 |

Nenhuma reta ultrapassa 75,7 % de exatidão no treino. A regularização L1 (λ = 0,005) torna a camada escondida mais esparsa sem alterar a exatidão.

## Acesso e edição

Qualquer pessoa pode **ver, correr o notebook e simular à vontade**; as simulações nunca alteram o repositório. Só o autor edita (licença [CC BY-NC-ND 4.0](LICENSE): partilha com crédito, uso não comercial, sem obras derivadas). A secção de referências (Minsky & Papert, 1969, e restantes) está no [`README.md`](README.md) e no fim da apresentação.
