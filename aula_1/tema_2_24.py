import marimo

__generated_with = "0.9.14"
app = marimo.App()

@app.cell()
def __(mo):
    import marimo as mo
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Introdução às Redes Neurais (Dia 02) ![Picture1.jpg](data:image/jpeg;base64,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)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Introdução (Exemplo 1)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        **Objetivo**: construir uma rede neural para classificar dígitos manuscritos da base de dados MNIST.

        * Os dígitos são imagens em escala de cinza com dimensão $28\times 28$ e distribuídas em 10 classes ($0,1,2,\ldots,9$).

        * Imagens em escala de cinza são matrizes e cada entrada dela equivale a um nível de intensidade luminosa. Neste conjunto de dados, as imagens variam a intensidade luminosa de $0$ (preto) a $255$ (branco) apenas com entradas inteiras.

        * Será então construída uma rede neural cuja entrada são as $784=28\times 28$ entradas das figuras e a saída é a classe em que esta imagem deve estar contida. Neste tipo de situação as matrizes que representam as imagens serão reescritas como um vetor, de tal forma que as entradas do vetor correspondem as colunas da matriz da imagem.

        * Esta rede terá duas camadas, além da entrada, com 10 neurônios, na primeira utilizaremos a função de ativação `ReLU` e na segunda camada a função `softmax`.

        * A classe numérica em que cada imagem está contida deverá ser convertida para o que se chama de `formato categórico`, isto é, a cada classe será associado um vetor em $\mathbb{R}^{10}.$ Por exemplo, a classe $0$ será associada ao vetor $(1,0,0,\ldots,0),$ a classe $1$ ao vetor $(0,1,0,\ldots,0)$ e assim por diante. Isto é necessário para podermos comparar adequadamente a saída da rede com a classe, pois estamos calculando distância entre vetores.

        * Aqui utilizaremos como função de perda a entropia cruzada (*cross-entropy)* $$\mathcal{L}(y,\hat{y}_\theta)=-\sum_{j=1}^{10} y_j\ln\hat{y}_{\theta,j},$$ sendo $y_j$ a classe original do dígito, $\hat{y}_{\theta,j}$ a classe estimada pela rede e $j$ é o índice das imagens utilizadas no treinamento.

        * A saída que será produzida pela rede neural é um vetor de probabilidades, uma vez que a função de ativação escolhida para a saída é a `softmax`. Isto é, dada uma imagem $\mathbf{x}$ de entrada, que deverá ser classificada, a rede produzirá como resposta, por exemplo, um vetor na forma $\hat{y}_{\theta,j}(\mathbf{x})=(0.05 \, ,\,0.07 \, ,\, 0.01\, ,\, 0.12\, ,\, 0.57\, ,\, 0.02\, ,\, 0.015\, ,\, 0.015\, ,\, 0.05\, ,\, 0.08).$ Nesta situação, a imagem $\mathbf{x}$ será classificada como um $4$.

        *OBS: o código base é de autoria de Samson Zhang (https://www.samsonzhang.com/building)*
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Parte 1: manipulação e visualização dos dados
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        **Iniciamos importando as bibliotecas necessárias!**

        `numpy` para cálculo numérico

        `panda` para manipulação de dados

        `pyplot` para imprimir as figuras
        """
    )
    return


@app.cell
def __():
    # bibliotecas básicas
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    return np, pd, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Aqui estamos lendo o arquivo que contém a informação das imagens.

        *obs: caso esteja utilizando o `google colab` espere carregar todo o arquivo `train.csv`, caso contrário a rede dará algum erro e não convergirá adequadamente.*
        """
    )
    return


@app.cell
def __(pd):
    # le o arquivo das imagens
    data = pd.read_csv('aula_03_(21_10)_train.csv')
    return (data,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Estamos lidando com 42000 amostras de imagens de manuscritos de tamanho $28\times 28=784$ já escritos no formato de vetor.

        Perceba que o vetor tem uma entrada a mais. Ela representa a etiqueta daquele número manuscrito.
        """
    )
    return


@app.cell
def __(data, np):
    data_1 = np.array(data)
    m, n = data_1.shape
    print(data_1.shape)
    return data_1, m, n


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Das 42000 imagens, separamos 4200 (10%) delas pra teste e as demais para o treino!
        """
    )
    return


@app.cell
def __(data_1, m, n, np):
    np.random.shuffle(data_1)
    data_dev = data_1[0:4200].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    data_train = data_1[4200:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    _, m_train = X_train.shape
    return X_dev, X_train, Y_dev, Y_train, data_dev, data_train, m_train


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Vamos analisar como são as imagens. Perceba que nas matrizes os valores são inteiros e variam entre $0$ e $255$.
        """
    )
    return


@app.cell
def __(X_train, m_train):
    #mostra as imagens
    Z = X_train.reshape((28,28,m_train)) # o método reshape serve para transformar o vetor empilhado de volta numa matriz

    print(Z[:,:,0]) # começamos mostrando como são os dados das imagens
    return (Z,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Agora vemos como os dígitos manuscritos são visualmente.
        """
    )
    return


@app.cell
def __(Y_train, Z, plt):
    fig, ax = plt.subplots(1, 1, figsize=(4,4))

    ax.imshow(Z[:,:,0],cmap='gray') # plota a imagem
    ax.set_title(f"Rótulo: {Y_train[0]}") # inclui o rótulo em que essa imagem está classificada
    ax.get_xaxis().set_visible(False) # tiram os eixos x e y da visualização
    ax.get_yaxis().set_visible(False)
    return ax, fig


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Aqui iremos dividir as entradas dos vetores de teste e treino por $255$, isto é, estamos fazendo o que é conhecido de normalização. Na prática isto não é uma normalização dos vetores das imagens, porém agora a escala de como a cor é vista irá variar entre $0$ e $1$, e não mais entre $0$ e $255$. Este tipo de cuidado é importante para ajudar a rede a convergir evitando que o gradiente tenha a possibilidade de crescer demais.
        """
    )
    return


@app.cell
def __(X_dev, X_train):
    X_dev_1 = X_dev / 255.0
    X_train_1 = X_train / 255.0
    return X_dev_1, X_train_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Parte 2: construção da rede neural
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Aqui definimos as funções de ativação que serão utilizadas: `ReLU` e `softmax`.

        Lembrando que:
        1. $\mathrm{ReLU}(x)=\max(0,x)$;
        2. $\frac{d}{dx} \mathrm{ReLU}(x)=1$, se $x>0$, e $\frac{d}{dx} \mathrm{ReLU}(x)=0$, se $x<0$;
        3. `softmax` produz um vetor em que cada entrada é dada por $\sigma(\mathbf{z})_j=\frac{\exp(z_j)}{\sum_k \exp(z_k)}.$
        """
    )
    return


@app.cell
def __(np):
    # definicao das funções de ativação
    def ReLU(Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(Z):
        return Z > 0 # retorna 1 se Z > 0  e retorna 0 se Z < 0

    def softmax(Z):
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return A
    return ReLU, ReLU_deriv, softmax


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Construímos aqui as operações nas camadas.

        1. Temos que $X$ é a matriz que contém as informações de cada imagem. Cada coluna desta matriz corresponde a uma imagem de um dígito. Essa matriz tem dimensão $784\times m$, $m$ para nós é a quantidade de imagens de treino. Neste caso, estamos usando $m=38800$.

        2. Daí, fazemos a transformação afim das entradas
        $Z^{[1]}=W^{[1]}X+b^{[1]},$
        sendo que a matriz $W^{[1]}$ contém as informações dos pesos que ligam as entradas aos  $10$ neurônios da primeira camada e $b^{[1]}$ corresponde aos viéses de cada neurônio da primeira cadama. O resultado dessa operação fornece um vetor $10\times m.$

        3. Em seguida, temos que aplicar a função de ativação para os neurônios da primeira camada fazendo $A^{[1]}=\mathrm{ReLU}(Z^{[1]}).$ Note que essa operação é feita entrada à entrada da matriz $Z^{[1]}$ e, portanto, a dimensão de $A^{[1]}$ é a mesma de $Z^{[1]}$.

        4. Para a segunda camada, onde será produzida a saída, repetimos o procedimento descrito no item 2. Fazemos novamente uma transformação afim das entradas. No entanto, aqui as entradas correspondem às saídas dos neurônios anteriores. Portanto, a matriz de pesos aqui precisa ter dimensão menor, no caso $10\times 10$. Assim, a transformação afim é $Z^{[2]}=W^{[2]}A^{[1]}+b^{[2]}.$ Novamente, aqui $Z^{[2]}$ produzirá uma matriz $10\times m.$

        5. Finalmente, precisamos aplicar a função de ativação na transformação afim que ocorreu na segunda camada. Neste caso a transformação será uma `softmax` que corresponderá as probabilidades citadas iniciais. Assim, $A^{[2]}=\mathrm{softmax}(Z^{[2]}).$
        """
    )
    return


@app.cell
def __(ReLU, softmax):
    # vamos operar as imagens nas camadas
    def forward_prop(W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2
    return (forward_prop,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Definindo os parâmetros iniciais: pesos e viéses

        O método `rand(a,b)` produz uma matriz de dimensões $a\times b$ com valores aleatórios que são amostrados da distribuição uniforme no intervalo $[0,1]$. Fazendo, por exemplo, `np.random.rand(10, 784) - 0.5` iremos gerar uma matriz $10\times 784$ com entradas no intervalo $[-0.5\, , \, 0.5].$

        Na prática a escolha inicial de pesos e viéses é feita usando método especializados. Tanto o Keras/TensorFlow como o Pytorch trazem as opções usuais de inicializadores para os pesos e viéses que variam desde as distribuições uniforme e normal até os inicializadores de Xavier Glorot. Este último, em particular, tem como objetivo inicializar os pesos de modo que a variância das ativações seja a mesma em todas as camadas. Essa variância constante ajuda a evitar que o gradiente exploda ou desapareça.

        *Referência: X. Glorot, Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, 2010. URL: https://proceedings.mlr.press/v9/glorot10a.html*
        """
    )
    return


@app.cell
def __(np):
    # definição dos parâmetros iniciais
    def init_params():
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2
    return (init_params,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Essa parte do código faz a codificação das classes das imagens para o formato categórico. Isto é importante para a comparação do resultado numérico gerado pela rede neural com os vetores de rótulos
        """
    )
    return


@app.cell
def __(np):
    # faz o one-hot encoding do vetor de rótulos
    def one_hot(Y): # one-hot encoding
        Y = Y.astype(int)
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    return (one_hot,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ### Parte 2.1: cálculo das derivadas e o *backpropagation*
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---

        Agora entramos na parte mais complicada que é o cálculo do gradiente para a atualização dos parâmetros.

        O que pretendemos calcular é a derivada da função de perda com relação a cada parâmetro de peso e viés. Para um classificador `softmax`, usaremos uma função de perda de entropia cruzada:
        $$\mathcal{L}(y,\hat{y}_\theta) = -\frac{1}{m}\sum_{i=1}^{10} y_i \ln(\hat{y}_{\theta,i}).$$
        *Obs: é praxe dividir a função de perda pela quantidade de amostras que estão sendo utilizadas, no caso aqui $m$.*

        Aqui, $\hat{y}_\theta$ é o vetor dos rótulos previstos e ele pode ter uma forma como:
        $$\begin{bmatrix} 0.01 \ 0.02 \ 0.05 \ 0.02 \ 0.80 \ 0.01 \ 0.01 \ 0.00 \ 0.01 \ 0.07\end{bmatrix}.$$

        O vetor $y$ é a codificação `one-hot` do rótulo do dado de treinamento. Se o rótulo para um exemplo de treinamento for 4, por exemplo, a sua codificação `one-hot` ficaria na forma:
        $$\begin{bmatrix} 0 \ 0 \ 0 \ 0 \ 1 \ 0 \ 0 \ 0 \ 0 \ 0 \ \end{bmatrix}.$$

        Observe que na soma $$\sum_{i=1}^{10} y_i \ln(\hat{y}_{\theta,i}),$$ o valor de $y_i = 0,$ para todos os $i$ exceto o rótulo correto. A função de perda para uma dada amostra,  é apenas o logaritmo da probabilidade dada para a previsão estimada. Em nosso exemplo acima, $$\mathcal{L}(y,\hat{y}_\theta) = -\ln(\hat{y}_5) = -\ln(0.80) \approx 0.2231.$$ Observe que, quanto mais próxima a probabilidade de predição estiver de 1, mais próxima a perda estará de 0. Conforme a probabilidade se aproxima de 0, a perda se aproxima de $+\infty$.

        A minimização da função de perda melhora a precisão do nosso modelo. Como foi possível ver no exemplo, quanto mais perto a rede aproxima dos rótulos originais, mais perto de zero ficará a função de perda. O processo de otimização é feito através da descida do gradiente, em que subtraímos dos parâmetros que queremos otimizar ($W^{[1]}, W^{[2]}, b^{[1]}$ e $b^{[2]}$) uma quantidade proporcional ao gradiente da função de perda naquela variável, isto é:
        $$W^{[1]} := W^{[1]} - \alpha \frac{\partial \mathcal{L}}{\partial W^{[1]}}$$
        $$ b^{[1]} := b^{[1]} - \alpha \frac{\partial \mathcal{L}}{\partial b^{[1]}}$$
        $$ W^{[2]} := W^{[2]} - \alpha \frac{\partial \mathcal{L}}{\partial W^{[2]}}$$
        $$ b^{[2]} := b^{[2]} - \alpha \frac{\partial \mathcal{L}}{\partial b^{[2]}}, $$
        o parâmetro $\alpha$ é o que chamamos de taxa de aprendizagem (*learning rate*).

        Nosso objetivo no *backpropagation* é calcular as derivadas $\frac{\partial \mathcal{L}}{\partial W^{[1]}},\frac{\partial \mathcal{L}}{\partial b^{[1]}},\frac{\partial \mathcal{L}}{\partial W^{[2]}}$ e $\frac{\partial \mathcal{L}}{\partial b^{[2]}}.$ Apenas para simplificar, escreveremos essas quantidades como $dW^{[1]}, db^{[1]}, dW^{[2]},$ e $db^{[2]}$. Esses valores são calculados usando a regra da cadeia retrocedendo em nossa rede, começando pelo cálculo de $\frac{\partial \mathcal{L}}{\partial Z^{[2]}}$, ou $dZ^{[2]}$. Não é imediato, como veremos a seguir, mas essa derivada é dada por:
        $$dZ^{[2]} = \frac{1}{m} \left(A^{[2]}-y\right).$$

        A partir desta derivada $dZ^{[2]}$, podemos utilizar a regra da cadeia, para $dW^{[2]}$ e $db^{[2]}$. Temos que:
        $dW^{[2]} = dZ^{[2]} A^{[1]T}$ e $db^{[2]} =  \Sigma {dZ^{[2]}}.$

        Então, para calcular $dW^{[1]}$ e $db^{[1]}$, primeiro encontraremos $dZ^{[1]}$, que é dado por:
        $$dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (Z^{[1]}).$$
        Em seguida, obtemos:
        $dW^{[1]} = dZ^{[1]} X^{T}$ e $db^{[1]} = \Sigma {dZ^{[1]}}.$
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Aqui vamos mostrar que a derivada $\frac{\partial \mathcal{L}}{\partial Z^{[2]}}=\frac{1}{m}\left(A^{[2]}-y\right).$

        Começamos, então, tirando o $\ln$ do `softmax` para obter:
        \begin{align*}\ln\hat{y}_{\theta,i} & =\ln\left(\frac{e^{z_{i}}}{\sum_{j=1}^{10}e^{z_{j}}}\right)\\
        & =\ln\left(e^{z_{i}}\right)-\ln\left(\sum_{j=1}^{10}e^{z_{j}}\right)\\
        & =z_{i}-\ln\left(\sum_{j=1}^{10}e^{z_{j}}\right)
        \end{align*}

        Em seguida derivamos a equação anterior em relação a variável $z_{k}$ para obtermos:
        \begin{align*}\frac{\partial\ln\hat{y}_{\theta,i}}{\partial z_{k}} & =\frac{\partial z_{i}}{\partial z_{k}}-\frac{\partial\ln\left(\sum_{j=1}^{10}e^{z_{j}}\right)}{\partial z_{k}}\end{align*}

        Note que o primeiro elemento de lado direito ficará na forma $$\frac{\partial z_{i}}{\partial z_{k}}=\begin{cases}
        1 & z_{i}=z_{k}\\
        0 & \text{caso contrário}
        \end{cases}=\mathbb{1}(z_{i}=z_{k})=\delta_{ik},$$ em que $\mathbb{1}(z_{i}=z_{k})$ é a função indicadora, que, neste caso, também pode ser representada pelo $\delta_{ik}$ delta de Kronecker.

        A segunda parcela do lado direito pode ser simplificada também, onde obtemos:
        \begin{align*}\frac{\partial\ln\left(\sum_{j=1}^{10}e^{z_{j}}\right)}{\partial z_{k}} & =\frac{1}{\sum_{j=1}^{10}e^{z_{j}}}\frac{\partial\left(\sum_{j=1}^{10}e^{z_{j}}\right)}{\partial z_{k}}\\
        & =\frac{1}{\sum_{j=1}^{10}e^{z_{j}}}\sum_{j=1}^{10}\frac{\partial e^{z_{j}}}{\partial z_{k}}\\
        & =\frac{1}{\sum_{j=1}^{10}e^{z_{j}}}\sum_{j=1}^{10}e^{z_{j}}\frac{\partial z_{j}}{\partial z_{k}}\\
        & =\frac{1}{\sum_{j=1}^{10}e^{z_{j}}}\sum_{j=1}^{10}e^{z_{j}}\mathbb{1}(z_{j}=z_{k})\\
        & =\frac{1}{\sum_{j=1}^{10}e^{z_{j}}}\sum_{j=1}^{10}e^{z_{j}}\delta_{jk}\\
        & =\frac{e^{z_{k}}}{\sum_{j=1}^{10}e^{z_{j}}}\\
        & =\hat{y}_{\theta,k}
        \end{align*}

        Concluímos então que:
        \begin{align*}\frac{\partial\ln\hat{y}_{\theta,i}}{\partial z_{k}} & =\delta_{ik}-\hat{y}_{\theta,k}\\
        \Rightarrow \frac{1}{\hat{y}_{\theta,i}}\frac{\partial\hat{y}_{\theta,i}}{\partial z_{k}} & =\delta_{ik}-\hat{y}_{\theta,k}\\
        \Rightarrow \frac{\partial\hat{y}_{\theta,i}}{\partial z_{k}} & =\hat{y}_{\theta,i}\left(\delta_{ik}-\hat{y}_{\theta,k}\right).
        \end{align*}


        Agora podemos diferenciar a entropia cruzada em relação a uma variável local $z_k$ do `softmax`.
        Sendo a entropia cruzada dada por
        $$\mathcal{L}(y,\hat{y}_{\theta})  =-\frac{1}{m}\sum_{i=1}^{10}y_{i}\ln\hat{y}_{\theta,i},$$
        então
        \begin{align*}
        \frac{\partial\mathcal{L}(y,\hat{y}_{\theta})}{\partial z_{k}} & =-\frac{1}{m}\sum_{i=1}^{10}y_{i}\frac{\partial\ln\hat{y}_{\theta,i}}{\partial z_{k}}\\
        & =-\frac{1}{m}\sum_{i=1}^{10}\frac{y_{i}}{\hat{y}_{\theta,i}}\frac{\partial\hat{y}_{\theta,i}}{\partial z_{k}}\\
        & =-\frac{1}{m}\sum_{i=1}^{10}\frac{y_{i}}{\hat{y}_{\theta,i}}\hat{y}_{\theta,i}\left(\delta_{ik}-\hat{y}_{\theta,k}\right)\\
        & =-\frac{1}{m}\sum_{i=1}^{10}y_{i}\left(\delta_{ik}-\hat{y}_{\theta,k}\right)\\
        & =-\frac{1}{m}\sum_{i=1}^{10}\left(y_{i}\delta_{ik}+y_{i}\hat{y}_{\theta,k}\right).
        \end{align*}
        Quando $i=k$ a primeira parcela do somatório anterior se tornará $y_{k}$, reduzindo a derivada da entropia cruzada a:
        \begin{align*}\frac{\partial\mathcal{L}(y,\hat{y}_{\theta})}{\partial z_{k}} & =\left(-y_{k}+\hat{y}_{\theta,k}\sum_{i=1}^{10}y_{i}\right).\end{align*}
        Observando que $\sum_{i=1}^{10}y_{i}=1$ já que $y$ é um vetor *one-hot*, obtemos então o resultado desejado:
        \begin{align*}
        \frac{\partial\mathcal{L}(y,\hat{y}_{\theta})}{\partial z_{k}} & =\frac{1}{m}\left(\hat{y}_{\theta,k}-y_{k}\right).
        \end{align*}

        Finalmente, na forma vetorizada (como será tratado pelo `numpy`), podemos simplesmente escrever o gradiente como
        \begin{align*}\frac{\partial\mathcal{L}(y,\hat{y}_{\theta})}{\partial z} & =\left(\hat{y}_{\theta}-y\right).\end{align*}

        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Finalmente, vamos mostrar como o $dW^{[2]}$ pode ser obtido.

        Temos que
        $$dW^{[2]}=\frac{\partial \mathcal{L}}{\partial W^{[2]}}=\frac{\partial \mathcal{L}}{\partial z^{[2]}}\frac{\partial z^{[2]}}{\partial W^{[2]}}=dZ^{[2]}\frac{\partial z^{[2]}}{\partial W^{[2]}}=dZ^{[2]}(A^{[1]})^{T}.$$

        As demais são obtidas de forma similar.
        """
    )
    return


@app.cell
def __(ReLU_deriv, m, np, one_hot):
    # cálculo das derivadas

    def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
        one_hot_Y = one_hot(Y)
        dZ2 = (1 / m) * (A2 - one_hot_Y)
        dW2 = dZ2.dot(A1.T)
        db2 = np.sum(dZ2, axis=1).reshape(-1, 1)
        dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = dZ1.dot(X.T)
        db1 = np.sum(dZ1, axis=1).reshape(-1, 1)
        return dW1, db1, dW2, db2
    return (backward_prop,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Atualiza os parâmetros de acordo com as derivadas. Faz a atualização de acordo com o gradiente descendente.
        """
    )
    return


@app.cell
def __():
    def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return (W1, b1, W2, b2)
    return (update_params,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ### Parte 2.2: definição da função que fará as épocas
        Efetivamente a função que faz o processo de cálculo da rede neural acontecer e atualiza os parâmetros sucessivamente
        """
    )
    return


@app.cell
def __(backward_prop, forward_prop, init_params, np, update_params):
    # vamos fazer a mágica acontecer!

    def get_predictions(A2):
        return np.argmax(A2, 0)

    def get_accuracy(predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(X, Y, alpha, iterations):
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 50 == 0:
                print("Iteração: ", i)
                predictions = get_predictions(A2)
                print(get_accuracy(predictions, Y))
        return W1, b1, W2, b2
    return get_accuracy, get_predictions, gradient_descent


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Parte 3: treinamento e análise dos resultados
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Finalmente realiza o treinamento!
        """
    )
    return


@app.cell
def __(X_train_1, Y_train, gradient_descent):
    W1, b1, W2, b2 = gradient_descent(X_train_1, Y_train, 0.1, 500)
    return W1, W2, b1, b2


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Funções para verificar a qualidade da nossa rede neural
        """
    )
    return


@app.cell
def __(X_dev_1, Y_dev, forward_prop, get_predictions, plt):
    def make_predictions(X, W1, b1, W2, b2):
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
        predictions = get_predictions(A2)
        return predictions

    def test_prediction(index, W1, b1, W2, b2):
        current_image = X_dev_1[:, index, None]
        prediction = make_predictions(X_dev_1[:, index, None], W1, b1, W2, b2)
        label = Y_dev[index]
        print('Previsão: ', prediction)
        print('Rótulo: ', label)
        current_image = current_image.reshape((28, 28))
        plt.imshow(current_image, cmap='gray')
        plt.axis('off')
        plt.show()
    return make_predictions, test_prediction


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Testa o quanto a nossa rede acerta!
        """
    )
    return


@app.cell
def __(W1, W2, X_dev_1, b1, b2, np, test_prediction):
    a, b = X_dev_1.shape
    test_prediction(np.random.randint(1, b), W1, b1, W2, b2)
    return a, b


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

