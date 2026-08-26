import pygame
from gomuku_engine import GomokuEngine
from config import load_cnn_model_params
from players import cnn_player

WIDTH, HEIGHT = 600, 600
BOARD_SIZE = 15
CELL_SIZE = WIDTH // BOARD_SIZE
BG_COLOR = (220, 179, 92)
LINE_COLOR = (0, 0, 0)


def draw_board(screen, engine):
    screen.fill(BG_COLOR)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (i * CELL_SIZE + CELL_SIZE // 2, CELL_SIZE // 2),
                         (i * CELL_SIZE + CELL_SIZE // 2, HEIGHT - CELL_SIZE // 2), 2)
        pygame.draw.line(screen, LINE_COLOR, (CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2),
                         (WIDTH - CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), 2)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = engine.board[r][c]
            center = (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2)
            if stone == 'X':
                pygame.draw.circle(screen, (0, 0, 0), center, CELL_SIZE // 2 - 4)
            elif stone == 'O':
                pygame.draw.circle(screen, (255, 255, 255), center, CELL_SIZE // 2 - 4)
    pygame.display.flip()
def bot(engine):
    bot = cnn_player(engine,**load_cnn_model_params())
    return bot


def play_with_gui(human_starts: bool, bot_fn):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gomoku: Human vs AI")
    engine = GomokuEngine()
    if human_starts:
        human_player = 'X'
        bot_player = 'O'
    else:
        human_player = 'O'
        bot_player = 'X'
    draw_board(screen, engine)
    while True:
        if engine.current_player == bot_player and not engine.is_game_over:
            r, c = bot_fn(engine)
            engine.make_move(r, c)
            draw_board(screen, engine)
            if engine.is_game_over:
                print(f"Game Finished! Winner: {engine.winner}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN and not engine.is_game_over:
                if engine.current_player == human_player:
                    x, y = event.pos
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE

                    if (row, col) in engine.get_legal_moves():
                        engine.make_move(row, col)
                        draw_board(screen, engine)
                        if engine.is_game_over:
                            print(f"Game Finished! Winner: {engine.winner}")