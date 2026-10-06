import unittest
import numpy as np
from PIL import Image
from tratamento_realista import compor


class Composicao(unittest.TestCase):
    def test_cabelo_progressivo_e_regioes_preservadas(self):
        original = np.full((200, 200, 3), (100, 65, 35), np.uint8)
        foto = Image.fromarray(original)
        cabelo = np.zeros((200, 200), np.float32)
        cabelo[10:50, 40:160] = 1
        pele = np.zeros_like(cabelo)
        pele[60:170, 50:150] = 1
        rosto = np.array([40, 40, 120, 130, 75, 90, 125, 90, 100, 115, 80, 140, 120, 140, .99])
        textura = original.copy()
        textura[65:150:4, 60:140] = (70, 50, 30)
        deltas = []
        for idade in (40, 50, 70, 90):
            saida = np.asarray(compor(foto, Image.fromarray(textura), (0, 0, 200, 200), pele, cabelo, rosto, idade))
            np.testing.assert_array_equal(saida[:, :30], original[:, :30])
            np.testing.assert_array_equal(saida[180:], original[180:])
            for x, y in [(75, 90), (125, 90), (100, 115), (100, 140)]:
                np.testing.assert_array_equal(saida[y-2:y+3, x-2:x+3], original[y-2:y+3, x-2:x+3])
            deltas.append(np.abs(saida[20:40, 60:140].astype(float)-original[20:40, 60:140]).sum())
        self.assertTrue(all(a < b for a, b in zip(deltas, deltas[1:])))

    def test_sem_mascara_nao_altera_foto(self):
        arr = np.full((200, 200, 3), (120, 80, 60), np.uint8)
        vazio = np.zeros((200, 200), np.float32)
        rosto = np.array([40, 40, 120, 130, 75, 90, 125, 90, 100, 115, 80, 140, 120, 140, .99])
        resultado = compor(Image.fromarray(arr), Image.new('RGB', (200, 200), 'black'),
                           (0, 0, 200, 200), vazio, vazio, rosto, 90)
        np.testing.assert_array_equal(np.asarray(resultado), arr)


if __name__ == '__main__':
    unittest.main()
