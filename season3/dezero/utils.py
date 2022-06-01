import os
import subprocess


# 변수 계산 그래프
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = "" if v.name is None else v.name

    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


# 함수 계산 그래프
def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    # 변수와 함수 간 edge
    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


# 계산 그래프 dot 언어로 표현
def get_dot_graph(output, verbose=True):
    txt = ""
    funcs = []
    seen_sets = set()

    def add_func(f):
        if f not in seen_sets:
            funcs.append(f)
            seen_sets.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        f = funcs.pop()
        txt += _dot_func(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n" + txt + "}"


# 계산 그래프 이미지 변환 및 대화 셀에 출력
def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose=True)

    tmp_dir = os.path.join(os.path.expanduser('~/Desktop/gitrepo/DeepLearningOnlyNumpy/season3'), 'dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as g:
        g.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    filename = os.path.join(tmp_dir, to_file)
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, filename)
    subprocess.run(cmd, shell=True)

    try:
        from IPython import display
        return display.Image(filename=filename)
    except:
        pass
