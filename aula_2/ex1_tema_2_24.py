import marimo

__generated_with = "0.9.14"
app = marimo.App()


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
        ## Introdução (Exemplo 2)

        **Objetivo:** resolver o mesmo problema de classificação de um dígito manuscrito, porém usando o Keras.

        Mas, o que é o Keras? Segundo sua própria definição é:

        "A `tf.keras` é uma API de alto nível para o TensorFlow, Pytorch e Jax para criar e treinar modelos de aprendizado profundo. Ela é usada para prototipagem rápida, pesquisa de ponta e produção, com três principais vantagens:

        * Fácil de usar:
        A Keras tem uma interface simples e consistente otimizada para os casos de uso comuns. Ela fornece feedback claro e prático para os erros do usuário.
        * Os modelos modulares e compostos:
        da Keras são feitos conectando elementos configuráveis, com poucas restrições.
        * Fácil de estender:
        Desenvolva elementos personalizados que expressem novas ideias para pesquisa. Crie novas camadas, métricas e funções de perda e desenvolva modelos de última geração."

        No site do Keras/Tensorflow é possível encontrar vários exemplos de problemas resolvidos: https://www.tensorflow.org/guide/keras?hl=pt-br.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Começamos da mesma forma, importando as bibliotecas necessárias para o trabalho. Perceba que além do `numpy` estamos importando métodos do `Keras`, `scikit-learn` e `seaborn`.
        """
    )
    return


@app.cell
def __():
    # importa o necessário
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.layers import Dense, Flatten
    from keras.models import Sequential
    from keras.utils import to_categorical
    from keras.datasets import mnist #enorme banco de dados de digitos manuscritos
    from sklearn.metrics import confusion_matrix # outra api de ml em python
    import seaborn as sns # biblioteca para visualizacao estatistica no python
    return (
        Dense,
        Flatten,
        Sequential,
        confusion_matrix,
        mnist,
        np,
        plt,
        sns,
        to_categorical,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        No `Keras` os dados do MNIST já estão presentes, então carregamos eles diretamente.
        """
    )
    return


@app.cell
def __(mnist):
    # carregar o banco de dados
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_test, X_train, y_test, y_train


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Vamos dar uma analisada nos dados.
        """
    )
    return


@app.cell
def __(X_test, X_train, y_test, y_train):
    # imprime o formato das matrizes
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Note que os dados de treino e teste já estão separados. Além disso, temos um conjunto bem maior de informação. São 70 mil imagens ao total.

        Outra informação relevante é que aqui as matrizes das imagens não estão empilhadas. Elas estão no seu formato original $28\times 28$.

        ---
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Aqui vamos apenas visualizar os dígitos manuscritos que estão presentes no conjunto de dados. É esperado que vamos ver figuras semelhantes aquelas que vimos anteriormente.
        """
    )
    return


@app.cell
def __(X_train, i, plt, y_train):
    num_classes = 10
    fig, ax = plt.subplots(1, num_classes, figsize=(20, 20))
    for _i in range(num_classes):
        sample = X_train[y_train == i][0]
        _ax[_i].imshow(sample, cmap='gray')
        _ax[_i].set_title(f'Rótulo: {_i}')
        _ax[_i].get_xaxis().set_visible(False)
        _ax[_i].get_yaxis().set_visible(False)
    return ax, fig, num_classes, sample


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        O `Keras` já possui um método que é capaz de transformar a informação numérica dos rótulos para o formato categórico.
        """
    )
    return


@app.cell
def __(np, to_categorical, y_test, y_train):
    temp = []
    for _i in range(len(y_train)):
        temp.append(to_categorical(y_train[_i], num_classes=10))
    y_train_1 = np.array(temp)
    temp = []
    for _i in range(len(y_test)):
        temp.append(to_categorical(y_test[_i], num_classes=10))
    y_test_1 = np.array(temp)
    print(y_train_1.shape)
    print(y_test_1.shape)
    return temp, y_test_1, y_train_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Vamos conferir agora se os dados precisam de alguma normalização.
        """
    )
    return


@app.cell
def __(X_train):
    # observem que os dados são matrizes 28x28 com valores inteiros entre 0 e 255
    X_train[0]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Precisa sim! Vamos fazê-la abaixo.
        """
    )
    return


@app.cell
def __(X_test, X_train):
    X_train_1 = X_train / 255
    X_test_1 = X_test / 255
    return X_test_1, X_train_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Finalmente, vamos utilizar a estrutura do `Keras` para construir a rede neural. Essa construção é através do método `Sequential()`. A partir da construção do modelo podemos adicionar as camadas que comporão a rede neural.
        """
    )
    return


@app.cell
def __(Dense, Flatten, Sequential):
    # cria a rede
    model = Sequential()
    model.add(Flatten(input_shape=(28,28))) # flatten serve para transformar a figura no seu formato original em um vetor empilhado
    model.add(Dense(128, activation='relu')) # dense são as camadas escondidas
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return (model,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        O método `summary()` vai mostras detalhes sobre essa rede construída.
        """
    )
    return


@app.cell
def __(model):
    # Mostra um resumo da rede construída
    model.summary()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Agora, com o método `compile()` vamos informar detalhes sobre a otimização. Por exemplo, qual tipo de função de perda, qual otimizador, qual a métrica de qualidade. Neste caso, estamos usando a entropia cruzada, o otimizador ADAM e a métrica é a acurácia.
        """
    )
    return


@app.cell
def __(model):
    # Define o otimizador, a função de perda (é o erro que se deseja minimizar) e a métrica de qualidade
    model.compile(loss='categorical_crossentropy',
    	      optimizer='adam',
    	      metrics=['acc'])
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Finalmente, vamos colocar o modelo para otimizar com o método `fit()`. Observem aqui que colocamos 10 épocas, isso é, 10 repetições entre as passagens pra frente e pra trás.
        """
    )
    return


@app.cell
def __(X_test_1, X_train_1, model, y_test_1, y_train_1):
    epochs = 10
    model.fit(X_train_1, y_train_1, epochs=epochs, validation_data=(X_test_1, y_test_1))
    return (epochs,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Com o modelo já calculado é hora de testar a sua qualidade. Foi possível perceber que ele forneceu quase 98% de eficiência na validação com os dados de teste. Isto é, ele deve errar pouco na hora de prever os dígitos.
        """
    )
    return


@app.cell
def __(X_test_1, model, np, plt):
    predictions = model.predict(X_test_1)
    predictions = np.argmax(predictions, axis=1)
    fig_1, axes = plt.subplots(ncols=10, sharex=False, sharey=True, figsize=(20, 4))
    for _i in range(10):
        random_num = np.random.choice(len(X_test_1))
        axes[_i].set_title(predictions[random_num])
        axes[_i].imshow(X_test_1[random_num], cmap='gray')
        axes[_i].get_xaxis().set_visible(False)
        axes[_i].get_yaxis().set_visible(False)
    plt.show()
    return axes, fig_1, predictions, random_num


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        Por fim, uma outra forma de visualizar os dados é através da matriz de confusão. Ela basicamente nos mostra os dígitos que foram classificados errados. Para gerar a matriz de confusão estamos utilizando o `seaborn`.
        """
    )
    return


@app.cell
def __(confusion_matrix, np, plt, predictions, sns, y_test_1):
    confusion_mtx = confusion_matrix(np.argmax(y_test_1, axis=1), predictions)
    fig_2, ax_1 = plt.subplots(figsize=(15, 10))
    _ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax_1, cmap='viridis')
    _ax.set_xlabel('Etiqueta prevista')
    _ax.set_ylabel('Etiqueta verdadeira')
    _ax.set_title('Matriz de Confusão')
    return ax_1, confusion_mtx, fig_2


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

