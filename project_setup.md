# Dependências

Para instalar as bibliotecas do Python necessárias para este projeto execute:

    pip install -r requirements.txt

# Coletando dados

Execute o programa `capture.py`:

    python capture.py

Para fechar aperte a tecla `ESC` e para capturar uma imagem aperte a tecla `c`.

Os dados são armazenados na pasta `data`. As imagens são arquivos no formato PNG e os metadados (localização da face e dos pontos detectados) estão disponíveis em um arquivo CSV. 

# Visualizando os dados

O programa `visualize.py` pode ser utilizado para visualizar os dados coletados. Assumindo que você queira visualizar os dados do arquivo `data-20180518-234757.csv` na pasta `data`:

    python visualize.py --data data/data-20180518-234757.csv

Pressione qualquer tecla para mostrar a próxima imagem e `ESC` para fechar o programa.
