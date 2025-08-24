import time
import random

# Definitions
colors = {
    "white":0, "black":1,
    0: "white", 1: "black"
}

pieces = {
    "e":0,
    "wp": 1, "wn": 2, "wb": 3, "wr": 4, "wq": 5, "wk": 6,
    "bp": 7, "bn": 8, "bb": 9, "br": 10, "bq": 11, "bk": 12
}

pvm = {
    pieces["wp"]: 100, pieces["wn"]: 290, pieces["wb"]: 320, pieces["wr"]: 500, pieces["wq"]: 920,
    pieces["bp"]: -100, pieces["bn"]: -290, pieces["bb"]: -320, pieces["br"]: -500, pieces["bq"]: -920,
}

pim = {
    pieces["e"]: " ",
    pieces["wp"]: "♙", pieces["wn"]: "♘", pieces["wb"]: "♗", pieces["wr"]: "♖", pieces["wq"]: "♕", pieces["wk"]: "♔",
    pieces["bp"]: "♟", pieces["bn"]: "♞", pieces["bb"]: "♝", pieces["br"]: "♜", pieces["bq"]: "♛", pieces["bk"]: "♚",
}

pst = {
    "p": (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    "n": ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    "b": ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    "r": (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    "q": (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    "k": (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}

def flip_pst_for_black(pst_white):
    pst_black = {}
    for piece, table in pst_white.items():
        flipped = [0] * 64
        for i in range(64):
            r, f = divmod(i, 8)
            mirror_i = (7 - r) * 8 + f
            flipped[mirror_i] = table[i]
        pst_black[piece] = flipped
    return pst_black

pst_black = flip_pst_for_black(pst)

piece_to_pst = {
    1: 'p', 7: 'p',   # pawns
    2: 'n', 8: 'n',   # knights
    3: 'b', 9: 'b',   # bishops
    4: 'r', 10: 'r',  # rooks
    5: 'q', 11: 'q',  # queens
    6: 'k', 12: 'k',  # kings
}

pst_strength = 0.7

rank_map = {2:"a", 3:"b", 4:"c", 5:"d", 6:"e", 7:"f", 8:"g", 9:"h"}
file_map = {2:"8", 3:"7", 4:"6", 5:"5", 6:"4", 7:"3", 8:"2", 9:"1"}

fen_piece_map = {
    'P': pieces["wp"], 'N': pieces["wn"], 'B': pieces["wb"], 'R': pieces["wr"], 'Q': pieces["wq"], 'K': pieces["wk"],
    'p': pieces["bp"], 'n': pieces["bn"], 'b': pieces["bb"], 'r': pieces["br"], 'q': pieces["bq"], 'k': pieces["bk"]
}

ob = []
for i in range(120):
    file = i % 10
    rank = i // 10
    if file < 2 or file > 9 or rank < 2 or rank > 9:
        ob.append(i)

def gpc(piece):
    return None if piece == pieces["e"] else (0 if piece <= 6 else 1)

def goc(color):
    return 1 - color
    
def index120_to_64(index120):
    rank = index120 // 10 - 2
    file = index120 % 10 - 2
    return rank * 8 + file

def index_to_pos(index):
    return rank_map[int(str(index)[1])] + file_map[int(str(index)[0])]

def cut_board_array(board_array):
    return [sq for i, sq in enumerate(board_array) if i not in ob]

def make_move(board, move):
    board_array = board.board_array
    new_board_array = board_array[:]  # copy

    sp, dp = move.sp, move.dp
    moving_piece = new_board_array[sp]
    moving_color = gpc(moving_piece)

    # --- en passant capture (remove the pawn that was passed) ---
    if moving_piece in (pieces["wp"], pieces["bp"]) and dp == board.en_pas_t and new_board_array[dp] == pieces["e"]:
        # captured pawn is behind the destination square (relative to mover)
        captured_sq = dp + (10 if moving_color == colors["white"] else -10)
        new_board_array[captured_sq] = pieces["e"]

    # --- basic move ---
    new_board_array[dp] = moving_piece
    new_board_array[sp] = pieces["e"]

    # --- promotion ---
    if move.promo is not None:
        new_board_array[dp] = move.promo

    # --- castling (move rook) ---
    if moving_piece in (pieces["wk"], pieces["bk"]) and abs(sp - dp) == 2:
        # kingside or queenside relative to 10x12
        if dp > sp:  # kingside
            rook_sp, rook_dp = sp + 3, sp + 1
        else:       # queenside
            rook_sp, rook_dp = sp - 4, sp - 1
        new_board_array[rook_dp] = new_board_array[rook_sp]
        new_board_array[rook_sp] = pieces["e"]

    # --- castling rights bookkeeping ---
    castling_rights = board.castling_rights.copy()

    # if a king moves, that side loses both rights
    if moving_piece == pieces["wk"]:
        castling_rights["wks"] = False
        castling_rights["wqs"] = False
    elif moving_piece == pieces["bk"]:
        castling_rights["bks"] = False
        castling_rights["bqs"] = False

    # if a rook moves off its original square, lose that side's corresponding right
    if sp == 99: castling_rights["wks"] = False   # white h1
    if sp == 92: castling_rights["wqs"] = False   # white a1
    if sp == 29: castling_rights["bks"] = False   # black h8
    if sp == 22: castling_rights["bqs"] = False   # black a8

    # if a rook is captured on its original square, lose that right too
    # (note: check the *destination* square after move, so look at the captured square in the original board)
    captured_piece = board_array[dp]
    if captured_piece == pieces["wr"]:
        if dp == 99: castling_rights["wks"] = False
        if dp == 92: castling_rights["wqs"] = False
    elif captured_piece == pieces["br"]:
        if dp == 29: castling_rights["bks"] = False
        if dp == 22: castling_rights["bqs"] = False

    # --- en passant target square for NEXT move ---
    # set only if a pawn just did a two-square push
    if moving_piece in (pieces["wp"], pieces["bp"]) and abs(dp - sp) == 20:
        en_pas_t = (sp + dp) // 2  # the square passed over
    else:
        en_pas_t = None

    # switch turn
    new_board_color = goc(board.current_color)

    return BoardState(
        new_board_array,
        current_color=new_board_color,
        castling_rights=castling_rights,
        en_pas_t=en_pas_t
    )

def default_board():
        board_array = [pieces["e"]] * 120

        # White pieces
        board_array[92] = pieces["wr"]
        board_array[93] = pieces["wn"]
        board_array[94] = pieces["wb"]
        board_array[95] = pieces["wq"]
        board_array[96] = pieces["wk"]
        board_array[97] = pieces["wb"]
        board_array[98] = pieces["wn"]
        board_array[99] = pieces["wr"]
        for file in range(2, 10):
            board_array[82 + file - 2] = pieces["wp"]

        # Black pieces
        board_array[22] = pieces["br"]
        board_array[23] = pieces["bn"]
        board_array[24] = pieces["bb"]
        board_array[25] = pieces["bq"]
        board_array[26] = pieces["bk"]
        board_array[27] = pieces["bb"]
        board_array[28] = pieces["bn"]
        board_array[29] = pieces["br"]
        for file in range(2, 10):
            board_array[32 + file - 2] = pieces["bp"]

        return BoardState(
            board_array,
            current_color=colors["white"],
            castling_rights={"wks": True, "wqs": True, "bks": True, "bqs": True},
            en_pas_t=None
        )

def random_position(move_count):
    logics = Logics()
    board = default_board()
    
    for i in range(move_count):
        legal_moves = logics.legal_moves(board)
        random_move = random.choice(legal_moves)
        
        board = make_move(board, random_move)
        
    return board

def fen_string_to_board(fen):
    board_array = [pieces["e"]] * 120
    parts = fen.split()
    board_part = parts[0]
    turn_part = parts[1]
    castling_part = parts[2]
    en_passant_part = parts[3]
    ranks = board_part.split('/')

    if len(ranks) != 8:
        raise ValueError("Invalid FEN")

    for r, rank in enumerate(ranks):
        file = 0
        for c in rank:
            if c.isdigit():
                file += int(c)
            else:
                index = (2 + r) * 10 + (2 + file)
                board_array[index] = fen_piece_map[c]
                file += 1

    current_color = colors["white"] if turn_part == 'w' else colors["black"]

    castling_rights = {"wks": False, "wqs": False, "bks": False, "bqs": False}
    if 'K' in castling_part:
        castling_rights["wks"] = True
    if 'Q' in castling_part:
        castling_rights["wqs"] = True
    if 'k' in castling_part:
        castling_rights["bks"] = True
    if 'q' in castling_part:
        castling_rights["bqs"] = True

    if en_passant_part != '-':
        file = ord(en_passant_part[0]) - ord('a') + 2
        rank = 10 - int(en_passant_part[1])
        en_pas_t = rank * 10 + file
    else:
        en_pas_t = None
    return BoardState(
        board_array,
        current_color=current_color,
        castling_rights=castling_rights,
        en_pas_t=en_pas_t
    )

class Move:
    def __init__(self, sp, dp, promo, is_capture):
        self.sp = sp
        self.dp = dp
        self.promo = promo
        self.is_capture = is_capture
        
    def get_uci(self):
        if self.promo == None:
            promo = ""
        else:
            promo = self.promo

        return index_to_pos(self.sp) + index_to_pos(self.dp) + promo
    
class BoardState:
    def __init__(self, board_array, current_color, castling_rights, en_pas_t):
        self.board_array = board_array
        self.current_color = current_color
        self.en_pas_t = en_pas_t

        if castling_rights == None:
            self.castling_rights = {"wks":False, "wqs":False, "bks":False, "bqs":False}
        else:
            self.castling_rights = castling_rights

class Logics:
    def pawn_moves(self, color, board, index, only_capture=False):
        moves = []
        A = board.board_array
        if color != gpc(A[index]):
            return []

        if color == colors["white"]:
            single, double, cl, cr = -10, -20, -11, -9
            start_rank, end_rank = 8, 2
            promos = [pieces["wq"], pieces["wr"], pieces["wb"], pieces["wn"]]
            ep_dir = -10
        else:
            single, double, cl, cr = 10, 20, 9, 11
            start_rank, end_rank = 3, 9
            promos = [pieces["bq"], pieces["br"], pieces["bb"], pieces["bn"]]
            ep_dir = 10

        # captures (left/right)
        for off in (cl, cr):
            to = index + off
            if to not in ob and A[to] != pieces["e"] and gpc(A[to]) == goc(color):
                r = to // 10
                if r == end_rank:
                    for p in promos:
                        moves.append(Move(index, to, p, True))
                else:
                    moves.append(Move(index, to, None, True))

        # en passant (target square is empty but capturable)
        if board.en_pas_t is not None:
            for to in (index + cl, index + cr):
                if to == board.en_pas_t and to not in ob:
                    moves.append(Move(index, to, None, True))

        if not only_capture:
            # single push
            to = index + single
            if to not in ob and A[to] == pieces["e"]:
                r = to // 10
                if r == end_rank:
                    for p in promos:
                        moves.append(Move(index, to, p, False))
                else:
                    moves.append(Move(index, to, None, False))
                # double push (only if single push was clear and on start rank)
                to2 = index + double
                if (index // 10) == start_rank and A[to2] == pieces["e"]:
                    moves.append(Move(index, to2, None, False))

        return moves

    def knight_moves(self, color, board, index):
        moves = []

        board_array = board.board_array

        if color != gpc(board_array[index]):
            return []

        knight_offsets = [-21, -19, -12, -8, 8, 12, 19, 21]


        for offset in knight_offsets:
            if index + offset not in ob:
                if board_array[index + offset] == pieces["e"] or gpc(board_array[index + offset]) == goc(color):
                    if gpc(board_array[index + offset]) == goc(color):
                        moves.append(Move(index, index + offset, None, True))
                    else:
                        moves.append(Move(index, index + offset, None, False))

        return moves
    
    def sliding_moves(self, color, board, index, offsets):
        moves = []
        for offset in offsets:
            dest = index + offset
            while dest not in ob:
                piece = board.board_array[dest]
                if piece == pieces["e"]:
                    moves.append(Move(index, dest, None, False))
                else:
                    if gpc(piece) == goc(color):
                        moves.append(Move(index, dest, None, True))
                    break
                dest += offset
        return moves

    def bishop_moves(self, c, b, i): return self.sliding_moves(c, b, i, [-11, -9, 9, 11])
    def rook_moves(self, c, b, i):   return self.sliding_moves(c, b, i, [-10, 10, -1, 1])
    def queen_moves(self, c, b, i):  return self.bishop_moves(c, b, i) + self.rook_moves(c, b, i)
    
    def is_index_attacked(self, index, board, attacker_color):
        board_array = board.board_array

        for sq_index, piece in enumerate(board_array):
            if sq_index in ob or piece == pieces["e"]:
                continue

            if gpc(piece) != attacker_color:
                continue

            if piece in [pieces["wp"], pieces["bp"]]:
                moves = self.pawn_moves(attacker_color, board, sq_index, only_capture=True)
            elif piece in [pieces["wn"], pieces["bn"]]:
                moves = self.knight_moves(attacker_color, board, sq_index)
            elif piece in [pieces["wb"], pieces["bb"]]:
                moves = self.bishop_moves(attacker_color, board, sq_index)
            elif piece in [pieces["wr"], pieces["br"]]:
                moves = self.rook_moves(attacker_color, board, sq_index)
            elif piece in [pieces["wq"], pieces["bq"]]:
                moves = self.queen_moves(attacker_color, board, sq_index)
            elif piece in [pieces["wk"], pieces["bk"]]:
                king_offsets = [-10, -9, -1, 1, 9, 10, 11, -11]
                moves = []
                for offset in king_offsets:
                    target = sq_index + offset
                    if target not in ob:
                        moves.append(Move(sq_index, target, None, False))
            else:
                moves = []

            for move in moves:
                if move.dp == index:
                    return True

        return False
    
    def king_moves(self, color, board, index):
        moves = []
        A = board.board_array
        if color != gpc(A[index]):
            return []

        castling_rights = board.castling_rights
        king_offsets = [-10, -9, -1, 1, 9, 10, 11, -11]

        # normal king steps (can’t step to an attacked square)
        for off in king_offsets:
            to = index + off
            if to not in ob and (A[to] == pieces["e"] or gpc(A[to]) == goc(color)):
                if not self.is_index_attacked(to, board, goc(color)):
                    moves.append(Move(index, to, None, A[to] != pieces["e"]))

        # castling: squares must be empty and NOT attacked: start, through, and destination
        if color == colors["white"] and index == 96:
            # kingside: e1->g1 (pass through f1=97)
            if castling_rights["wks"] and A[97] == pieces["e"] and A[98] == pieces["e"] and A[99] == pieces["wr"]:
                if not self.is_index_attacked(96, board, colors["black"]) and \
                   not self.is_index_attacked(97, board, colors["black"]) and \
                   not self.is_index_attacked(98, board, colors["black"]):
                    moves.append(Move(index, 98, None, False))
            # queenside: e1->c1 (pass through d1=95)
            if castling_rights["wqs"] and A[95] == pieces["e"] and A[94] == pieces["e"] and A[93] == pieces["e"] and A[92] == pieces["wr"]:
                if not self.is_index_attacked(96, board, colors["black"]) and \
                   not self.is_index_attacked(95, board, colors["black"]) and \
                   not self.is_index_attacked(94, board, colors["black"]):
                    moves.append(Move(index, 94, None, False))

        if color == colors["black"] and index == 26:
            # kingside: e8->g8 (f8=27)
            if castling_rights["bks"] and A[27] == pieces["e"] and A[28] == pieces["e"] and A[29] == pieces["br"]:
                if not self.is_index_attacked(26, board, colors["white"]) and \
                   not self.is_index_attacked(27, board, colors["white"]) and \
                   not self.is_index_attacked(28, board, colors["white"]):
                    moves.append(Move(index, 28, None, False))
            # queenside: e8->c8 (d8=25)
            if castling_rights["bqs"] and A[25] == pieces["e"] and A[24] == pieces["e"] and A[23] == pieces["e"] and A[22] == pieces["br"]:
                if not self.is_index_attacked(26, board, colors["white"]) and \
                   not self.is_index_attacked(25, board, colors["white"]) and \
                   not self.is_index_attacked(24, board, colors["white"]):
                    moves.append(Move(index, 24, None, False))

        return moves

    def legal_moves(self, board):
        board_array = board.board_array
        current_color = board.current_color

        # Find king position
        king_square = None
        for i, p in enumerate(board_array):
            if current_color == colors["white"] and p == pieces["wk"]:
                king_square = i
                break
            elif current_color == colors["black"] and p == pieces["bk"]:
                king_square = i
                break

        # Generate all pseudo-legal moves
        pseudo_moves = []
        for sq_index, piece in enumerate(board_array):
            if sq_index in ob or piece == pieces["e"]:
                continue
            if gpc(piece) != current_color:
                continue

            if piece in [pieces["wp"], pieces["bp"]]:
                pseudo_moves.extend(self.pawn_moves(current_color, board, sq_index))
            elif piece in [pieces["wn"], pieces["bn"]]:
                pseudo_moves.extend(self.knight_moves(current_color, board, sq_index))
            elif piece in [pieces["wb"], pieces["bb"]]:
                pseudo_moves.extend(self.bishop_moves(current_color, board, sq_index))
            elif piece in [pieces["wr"], pieces["br"]]:
                pseudo_moves.extend(self.rook_moves(current_color, board, sq_index))
            elif piece in [pieces["wq"], pieces["bq"]]:
                pseudo_moves.extend(self.queen_moves(current_color, board, sq_index))
            elif piece in [pieces["wk"], pieces["bk"]]:
                pseudo_moves.extend(self.king_moves(current_color, board, sq_index))

        # Filter out moves that leave king in check
        legal_moves = []
        for move in pseudo_moves:
            new_board = make_move(board, move)

            # Update king square if the king moved
            new_king_square = king_square
            if move.sp == king_square:
                new_king_square = move.dp

            if not self.is_index_attacked(new_king_square, new_board, goc(current_color)):
                legal_moves.append(move)
        
        # Filter out off board moves
        legal_moves = [m for m in legal_moves if m.sp not in ob]
        legal_moves = [m for m in legal_moves if m.dp not in ob]
        return legal_moves
    
class Engine:
    def __init__(self):
        self.logics = Logics()
        self.show_thought = True

    def is_in_check(self, board, color):
        # find king
        ks = None
        for i, p in enumerate(board.board_array):
            if p == (pieces["wk"] if color == colors["white"] else pieces["bk"]):
                ks = i
                break
        return self.logics.is_index_attacked(ks, board, goc(color))

    def is_checkmate(self, board):
        # no legal moves AND king is in check
        if self.logics.legal_moves(board):
            return False
        return self.is_in_check(board, board.current_color)
    
    def eval(self, board):
        eval_score = 0
        
        for sq64 in range(64):
            piece = cut_board_array(board.board_array)[sq64]

            if piece == pieces["e"]:
                continue

            ptype = piece_to_pst[piece]

            if piece not in [pieces["wk"], pieces["bk"]]:
                eval_score += pvm[piece]
                if gpc(piece) == colors["white"]:
                    eval_score += int(pst[ptype][sq64] * pst_strength)
                elif gpc(piece) == colors["black"]:
                    eval_score -= int(pst_black[ptype][sq64] * pst_strength)

        return eval_score
    
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.is_checkmate(board):
            return self.eval(board), None

        legal_moves = self.logics.legal_moves(board)
        if not legal_moves:
            return self.eval(board), None

        best_move = None

        if maximizing_player:
            max_eval = float("-inf")
            for move in legal_moves:
                new_board = make_move(board, move)
                eval_score, _ = self.minimax(new_board, depth - 1, alpha, beta, False)

                if self.show_thought:
                    print(f"\r[Depth {depth}] Trying {move.get_uci()} -> Eval: {eval_score} \033[K", end="")

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    if self.show_thought:
                        print(f"\r[Depth {depth}] Beta cutoff at move {move.get_uci()} \033[K", end="")
                    break  # beta cut off
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in legal_moves:
                new_board = make_move(board, move)
                eval_score, _ = self.minimax(new_board, depth - 1, alpha, beta, True)

                if self.show_thought:
                    print(f"\r[Depth {depth}] Trying {move.get_uci()} -> Eval: {eval_score} \033[K", end="")

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    if self.show_thought:
                        print(f"\r[Depth {depth}] Alpha cutoff at move {move.get_uci()} \033[K", end="")
                    break  # alpha cut off
            return min_eval, best_move
        
    def search_best_move(self, for_color, board, depth):
        print("\033[?25l", end="")

        _, best_move = self.minimax(
            board,
            depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing_player=(for_color == colors["white"])
        )

        print("\033[?25h", end="")

        return best_move

class CLI:
    def __init__(self):
        self.version = "1.0.0"

        self.play_as = colors["white"]

        self.engine = Engine()

        # Settings
        self.search_depth = 3
        self.show_thought = True
        self.show_legal_moves = False
        self.show_eval = True
        self.show_board = True

    def give_options(self, options, returner=None):
        for key, option in options.items():
            print(f"({key}) {option[0]}")
        
        while True:
            chosen = input("> ")
            chosen = chosen.replace(" ", "").lower()

            if chosen not in options:
                print("\n\033[31mOption not found\033[0m\n")
            else:
                print()
                options[chosen][1]()

                # Go back to returner
                if returner != None:
                    print()
                    returner()

                return chosen

    def start(self):
        print("\x1b[44m \033[1mDepth\033[0m\x1b[44m - Python Chess Engine \x1b[0m")
        print(f"\u001b[46m Version: {self.version} \u001b[0m")

        self.give_options({
            "a":("About", self.about),
            "c":("Configure", self.configure),
            "n":("Analyze", self.analyze),
            "p":("Play", self.play),
            "q":("Quit", quit)
        }, returner=self.start)

    def about(self):
        print("\u001b[41m About \u001b[0m")
        print("Depth is a chess engine that was originally going to be written in Javascript.")
        print("However, I used Python because I was more experienced with it.")
        print("It will be slow since it is python.")

    def configure(self, return_to_play=False):
        print("\u001b[42m Configure \u001b[0m")
        def change_search_depth():
            new_depth = input("New depth > ")
            self.search_depth = int(new_depth)
        
        def change_show_thought():
            show_thougth = input("t/f > ")
            if show_thougth.lower() == "t":
                self.show_thought = True
            else:
                self.show_thought = False

        def change_show_legal_moves():
            show_legal_moves = input("t/f > ")
            if show_legal_moves.lower() == "t":
                self.show_legal_moves = True
            else:
                self.show_legal_moves = False
        
        def change_show_board():
            show_board = input("t/f > ")
            if show_board.lower() == "t":
                self.show_board = True
            else:
                self.show_board = False

        self.give_options({
                "1": (f"Search Depth: {self.search_depth}", change_search_depth),
                "2": (f"Show thought: {self.show_thought}", change_show_thought),
                "3": (f"Show legal moves: {self.show_legal_moves}", change_show_legal_moves),
                "4": (f"Show board: {self.show_board}", change_show_board),
                "e": ("Exit", self.start)
        }, returner=self.configure)

    def analyze(self):
        print("\u001b[45m Analyze \u001b[0m")
        print("G to generate random board")
        print("Enter FEN string to analyze (E to exit)")
        fen_string = input("> ")
            
        if fen_string.lower() == "e":
            print()
            self.start()

        if fen_string.lower() == "g":
            print()
            print("Move count <= 100")
            move_count = int(input("> "))
            if move_count <= 100:
                board = random_position(move_count)
            else:
                return
        else:
            board = fen_string_to_board(fen_string)

        if self.show_board:
            print()
            self.print_board(board)

        # Count captured pieces
        starting_counts = {
            pieces["wp"]: 8, pieces["wn"]: 2, pieces["wb"]: 2, pieces["wr"]: 2, pieces["wq"]: 1,
            pieces["bp"]: 8, pieces["bn"]: 2, pieces["bb"]: 2, pieces["br"]: 2, pieces["bq"]: 1
        }

        current_counts = {p:0 for p in starting_counts}

        for sq in board.board_array:
            if sq in current_counts:
                current_counts[sq] += 1

        captured_white = []
        captured_black = []

        for piece, start_count in starting_counts.items():
            captured = start_count - current_counts[piece]
            if captured > 0:
                if piece <= 6:  # white piece
                    captured_black.extend([piece] * captured)
                else:            # black piece
                    captured_white.extend([piece] * captured)

        print()
        print("Captured by White: ", " ".join(pim[p] for p in captured_white) or "None")
        print("Captured by Black: ", " ".join(pim[p] for p in captured_black) or "None")

        # Evaluate position
        eval_score = self.engine.eval(board)
        print(f"Evaluation: {eval_score}")

        # Evaluation bar
        max_bar_length = 20
        scale = 100
        bar_length = int(min(abs(eval_score) / scale, max_bar_length))

        if eval_score > 0:
            bar = "█" * bar_length + "-" * (max_bar_length - bar_length)
            print(f"[{bar}] White is better")
        elif eval_score < 0:
            bar = "█" * bar_length + "-" * (max_bar_length - bar_length)
            print(f"[{bar}] Black is better")
        else:
            bar = "-" * max_bar_length
            print(f"[{bar}] Equal")

        print()
        color = board.current_color

        start_time = time.time()
        print(f"Search depth: {self.search_depth}")
        best_move = self.engine.search_best_move(color, board, self.search_depth)
        elapsed = time.time() - start_time
        print()
        print(f"Best for {colors[color]}: {best_move.get_uci()} | Thought for {elapsed:.2f}s")

    def print_board(self, board):
        board_array = board.board_array

        if self.play_as == colors["white"]:
            print("  a b c d e f g h")
            for rank in range(2, 10):
                row = []
                for file in range(2, 10):
                    index = rank * 10 + file
                    row.append(pim.get(board_array[index], "?"))
                print(f"{rank - 1} " + " ".join(row) + f" {rank - 1}")
            print("  a b c d e f g h")

        elif self.play_as == colors["black"]:
            print("  h g f e d c b a")
            for rank in range(9, 1, -1):
                row = []
                for file in range(9, 1, -1):
                    index = rank * 10 + file
                    row.append(pim.get(board_array[index], "?"))
                print(f"{10 - rank} " + " ".join(row) + f" {10 - rank}")
            print("  h g f e d c b a")

    def play(self):
        print("\u001b[43m Play \u001b[0m")
        print(f"\u001b[1mCurrent configuration\u001b[0m")
        print(f"Search depth: {self.search_depth}")
        print("E to exit\n")
        
        def set_white():
            self.play_as = colors["white"]
            
        def set_black():
            self.play_as = colors["black"]

        self.give_options({
            "1":("Play as white", set_white),
            "2":("Play as black", set_black),
            "e":("Exit", self.start)
        })
        print(f"\u001b[43m Playing as {colors[self.play_as].upper()} \u001b[0m")

        current_board = default_board()

        while True:
            if self.show_board:
                print()
                self.print_board(current_board)

            if current_board.current_color == self.play_as:
                legal_moves = self.engine.logics.legal_moves(current_board)
                if not legal_moves:
                    print("\u001b[41m Checkmate or stalemate, game over \u001b[0m")
                    break
                
                if self.show_eval:
                    print()
                    print(f"Evaluation: {self.engine.eval(current_board)}")
                
                if self.show_legal_moves:
                    for move in legal_moves:
                        print(move.get_uci())
                print(f"{len(legal_moves)} Legal moves found\n")

                move_input = input("\u001b[41mUCI\u001b[0m > ").strip().lower()
                
                if move_input == "e":
                    print()
                    self.start()
                
                move_found = None
                for move in legal_moves:
                    if move.get_uci() == move_input:
                        move_found = move
                        break
                
                if move_found:
                    current_board = make_move(current_board, move_found)
                else:
                    print("\n\033[31mInvalid move, try again\033[0m")
            else:
                print()
                print(f"Evaluation: {self.engine.eval(current_board)}")
                print("Engine thinking...")
                
                start_time = time.time()
                best_move = self.engine.search_best_move(current_board.current_color, current_board, self.search_depth)
                elapsed = time.time() - start_time
                if best_move is None:
                    print("\u001b[41m Engine has no moves, game over \u001b[0m")
                    break
                print("\n")
                print(f"\x1b[44m Engine plays: {best_move.get_uci()} \x1b[0m | Thought for {elapsed:.2f}s")
                current_board = make_move(current_board, best_move)

if __name__ == "__main__":
    app = CLI()
    app.start()