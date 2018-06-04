# Entendendo o Dataset

### Linhas -> Imagens
### Colunas 
1 - nome da imagem com data e horário
2 - face_x = X onde começa o retângulo verde da imagem
3 - face_y = Y onde começa o retângulo verde da imagem
4 - face_width = diferença entre Xinicial e Xfinal do retângulo verde
5 - face_height = diferença entre Yinicial e Yfinal do retângulo verde
Xn = Posição X do ponto n
Yn = Posição Y do ponto n

Precisamos criar uma maneira de utilizar o capture.py em imagens que não sejam provenientes da WebCam para usar base de dados de imagens de diferentes emoções

Os nossos pontos vão de 0 a 67, tendo a mesma ditribuição dos pontos dados pela autora do Artigo no qual nos baseamos. A foto que explica a distribuiçãodos pontos está na mesma pasta que os códigos e se chama face_dots.png

CITAR 

- Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
- Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.
