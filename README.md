# capstone_udacity
Capstone Project Udacity - Time Series Forecasting

Análise Exploratória de Dados usando Python - Shanghai license plate buildding price prediction
================

Material desenvolvido como complemento.

Preparação do Ambiente de Desenvolvimento
-----------------------------------------

Para instalar o **Python** e as dependendências necessárias, execute os seguintes passos:

1.  Instale o [Anaconda](https://www.continuum.io/downloads) ou o [Miniconda](https://conda.io/miniconda.html) (versão reduzida) de acordo com a versão do Python de sua preferência, `python2` ou `python3`. Reinicie o sistema operacional após a instalação.

        jupyter notebook


# Machine Learning Capstone Project
## Nanodegree Engenheiro de Machine Learning


### Marcus Vinicius de Oliveira Cruz
### 21 de Março de 2018








### Time Series Forescasting - Shanghai license plate buildding price prediction










## Project Overview


Como conclusão do nanodegree Engenheiro de Machine Learning e por uma possibilidade de uma consultoria de machine learning resolvi ir a fundo em um modelo de time series forecasting. Um modelo de série temporal basicamente consiste em um modelo estatístico que analisa uma variação temporal e consegue realizar previsões.
Escolhi o dataset Shanghai license plate bidding price prediction para construir o meu modelo do capstone. Esse é um dataset que pertence ao Kaggle.¶

https://www.kaggle.com/bogof666/shanghai-car-license-plate-auction-price
Definições principais

O aumento da propriedade e uso de automóveis na China nas últimas duas décadas aumentou o consumo de energia, piora a poluição do ar e congestionamento exacerbado. O governo de Xangai adotou um sistema de leilão para limitar o número de placas emitidas para cada mês. O conjunto de dados contém dados históricos de leilões de janeiro de 2002 a outubro de 2017.
como funciona o sistema de leilão: um preço inicial é dado no início do leilão, os licitantes só podem oferecer até 3 vezes por cada leilão e só podem marcar para cima ou para baixo dentro de 300 CNY (aproximadamente 46 USD) por cada lance. No final de cada leilão, apenas o n superior (número de placas que serão emitidas para o mês) receberá as placas de licença ao custo de suas propostas. A oferta n. ° será o preço mais baixo do mês. Por favor, note que os leilões são realizados on-line e cada licitante não poderá ver outros lances.
