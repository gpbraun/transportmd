# TransportMD

## Instalação

A maneira mais simples de ter acesso a todas as funcionalidades da API de Python para LAMMPS é instalando o pacote usando `conda`.

```shell
conda install conda-forge::lammps
```

Para instalar o pacote com o `poetry` é necessário configurar...

```shell
poetry config virtualenvs.path $CONDA_ENV_PATH
poetry config virtualenvs.create false
```

