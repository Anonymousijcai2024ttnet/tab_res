import os
import numpy as np


def load_expression(p, nmax=5):
    """Load the expression as a list of clauses of literals

        ex:

        if dnf = (x_0 & x_1) | (x_1 & x_4) | (x_2 & x_4) | (x_0 & ~x_2)
        returns [['x_0', 'x_1'], ['x_1', 'x_4'], ['x_2', 'x_4'], ['x_0', '~x_2']]"""

    with open(p, 'r') as file:
        lits = file.readline()
        lits = lits.replace(' ', '')
        lits = lits.replace('(', '')
        lits = lits.replace(')', '')

    name_file = os.path.basename(p)

    if "CNF" in name_file:
        clause_ = lits.split('&')
        expression = []
        for c in clause_:
            expression.append(c.split('|'))
    else:  # DNF
        clause_ = lits.split('|')
        expression = []
        for c in clause_:
            expression.append(c.split('&'))
    if nmax is not None:
        expression = [i for i in expression if len(i) <= nmax]
    return expression


def insertion_sort(inputlist):
    for i in range(1, len(inputlist)):
        j = i - 1
        nxt_element = inputlist[i]
        # Compare the current element with next one
        while (int(inputlist[j][-1]) > int(nxt_element[-1])) and (j >= 0):
            inputlist[j + 1] = inputlist[j]
            j = j - 1
            inputlist[j + 1] = nxt_element


def map_lit_to_idx(expression):
    """Change the variables x0 ... x_i to match the number of columns nlit

    If all the clauses have 2 literals but the number of literals is high, it would create an issue when computing the
    lookup table. We want to keep the lookup table as small as possible. Therefore, we can first compute the truth table
    with as many literals as we need as numpy is not computationnally expensive. But we have to be sure that the lookup
    table stay as small as possible"""

    all_variables = []
    for c in expression:
        for x in c:
            if x.startswith('~'):
                all_variables.append(x[1:])
            else:
                all_variables.append(x)
    all_variables = list(set(all_variables))
    insertion_sort(all_variables)
    var_to_col = {var: i for i, var in enumerate(all_variables)}
    return var_to_col


def clause2table(c, nlit=5, op='|', mapping=None):
    """Create the truth table from a clause of a cnf or dnf, with a maximum of
    nlit literals."""
    # n = len(c)
    if mapping is None:
        mapping = {var: i for i, var in enumerate(list(range(5)))}
    insertion_sort(c)

    # create all combinations of the nmax literals
    cols = []
    for col in range(nlit):
        cols.append([np.ceil(i / 2 ** (nlit - col - 1)) % 2 for i in range(2 ** nlit)])
    cols = np.array(cols).transpose()
    cols = np.roll(cols, shift=2 ** (nlit - 1) - 1, axis=0)
    cols = np.flip(cols, axis=0)
    cols[:, 0] = np.flip(cols[:, 0])
    cols = (cols == 1)

    # apply the expression on the literals to get the truth table
    if op == '|':
        res = np.zeros((2 ** nlit)) == 1  # cnf so False | res = res
        for x in c:
            if x.startswith('~'):
                i = mapping[x[1:]]
                res = np.logical_or(~cols[:, i], res)
            else:
                i = mapping[x]
                res = np.logical_or(cols[:, i], res)
    else:
        res = np.ones((2 ** nlit)) == 1  # dnf so True & res = res
        for x in c:
            if '8' in x:
                continue
            if x.startswith('~'):
                i = mapping[x[1:]]
                res = np.logical_and(~cols[:, i], res)

            else:
                i = mapping[x]
                res = np.logical_and(cols[:, i], res)

    return res


def expression2table(expression, nmax=5, op='|'):
    """Create the truth table of an expression cnf or dnf"""
    # mapping = map_lit_to_idx(expression, nmax)
    mapping = {f'x_{i}': i for i, var in enumerate(list(range(nmax)))}
    tt_clauses = []
    for clause in expression:
        tt_clauses.append(clause2table(clause, nmax, op, mapping))

    if op == '|':  # CNF
        tt_expr = np.ones((2 ** nmax)) == 1
        for clause in expression:
            tt_clause = clause2table(clause, nmax, op, mapping)
            tt_expr = np.logical_and(tt_expr, tt_clause)

    else:  # DNF
        tt_expr = np.zeros((2 ** nmax)) == 1
        for clause in expression:
            tt_clause = clause2table(clause, nmax, op, mapping)
            tt_expr = np.logical_or(tt_expr, tt_clause)

    return tt_expr


def create_expr_for_folder(expr_files, npatches_per_filter, exception=None):
    """Order all the expressions for the right patches

    Ex:
    The file 'DNF_expression_block0_filter_0_9_coefdefault_1.0_sousblock_None.txt' must be in the 9th position
    in the filter 0 and the file 'DNF_expression_block0_filter_0_coefdefault_1.0_sousblock_None.txt' in all other
    positions for the filter 0"""
    if exception is None:
        exception = [2]
    expr_files = sorted(expr_files)
    # order the files for each block
    filters = [f.split('_')[4] for f in expr_files]
    filters = list(set(filters))
    block = {int(i): [] for i in filters if int(i) not in exception}
    # removing exception blocks
    for file in expr_files:
        if int(file.split('_')[4]) in exception:
            continue
        block[int(file.split('_')[4])].append(file)
        # print(file)

    size = max([len(f) for f in expr_files])
    all_expr = np.zeros((len(filters) - len(exception), npatches_per_filter), dtype=f'a{size}')

    idx = 0
    # putting the right expressions at each position
    for i in range(len(filters)):
        if i in exception:
            continue
        # print(i,idx)
        block[i].sort()
        all_expr[idx, :] = block[i][-1]
        for e in block[i]:
            # print(e)
            # print(e, e.split('_')[5])
            if e.split('_')[5] == 'None' or e.split('_')[5] == 'coefdefault':
                continue
            else:
                all_expr[idx, int(e.split('_')[5])] = e
        idx += 1

    return all_expr


def load_table_as_np(folder, npatches_per_filter, nmax, op, exceptions):
    expr_files = os.listdir(folder)
    expr = []
    for e in expr_files:
        if op == '|' or op.lower() == 'cnf':  # CNF
            if 'CNF' in e:
                expr.append(e)
        else:  # DNF
            if 'DNF' in e:
                expr.append(e)
    expr_files = expr
    expr_files.sort()
    all_expr = create_expr_for_folder(expr_files, npatches_per_filter, exceptions)

    all_expr = all_expr.flatten().tolist()
    lookups = []
    for file in all_expr:
        fpath = os.path.join(folder, file.decode())
        expression = load_expression(fpath, nmax)
        table = expression2table(expression, nmax, op)
        lookups.append(table.astype(np.uint8))
    return lookups


def load_table_as_np_inference(folder, nmax, op, exceptions):
    expr_files = os.listdir(folder)
    expr = []
    for e in expr_files:
        if op == '|' or op.lower() == 'cnf':  # CNF
            if 'CNF' in e:
                expr.append(e)
        else:  # DNF
            if 'DNF' in e:
                expr.append(e)
    expr_files = expr
    expr_files.sort()
    all_expr = []
    for f in expr_files:
        if int(f.split('_')[2].replace('block', '')) in exceptions:
            continue
        if f.split('_')[5] == 'None':
            all_expr.append(f)
    lookups = []
    for file in all_expr:
        fpath = os.path.join(folder, file)
        expression = load_expression(fpath, nmax)
        table = expression2table(expression, nmax, op)
        lookups.append(table.astype(np.uint8))
    return lookups


def get_lookup_tables(folder, patches_nb, nlit, op, num_blocks=2, exceptions=None):
    """Create the lookup tables for each block"""

    if exceptions is None:
        exceptions = []
    if op == '|':
        expr_type = "CNF"
    else:
        expr_type = "DNF"
    tables = []
    nexpressions = []
    for j in range(num_blocks):
        block_num = str(j)
        files = os.listdir(folder)
        block = f'block{block_num}'
        npatches_per_filter = patches_nb[j]
        nmax = nlit[j]
        expr_files = [f for f in files if expr_type in f and block in f]
        all_expr = create_expr_for_folder(expr_files, npatches_per_filter, exception=exceptions)
        all_expr = all_expr.flatten().tolist()
        lookups = []
        for i, file in enumerate(all_expr):
            fpath = os.path.join(folder, file.decode())
            expression = load_expression(fpath, nmax)
            table = expression2table(expression, nmax, op)
            lookups.append(table)
        multi_lookup = [lk for lk in lookups]
        # print(f"Table {j} size : {len(lookups)}")
        tables.append(multi_lookup)
        nexpressions.append(len(all_expr))

    return tables, nexpressions


if __name__ == '__main__':
    print()
