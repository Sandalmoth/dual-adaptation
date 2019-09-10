configfile:
    "config.yml"


rule parametrize:
    input:
        up = "data/raw_internal/{upfile}.csv",
        down = "data/raw_internal/{downfile}.csv",
    output:
        "intermediate/{upfile}.{downfile}.db"
    shell:
        """
        python3 code/abc.py parametrize \
            -p simulation-config.toml \
            -u {input.up} \
            -d {input.down} \
            -b {output}
        """


rule make_verify_input:
    input:
        "intermediate/{name}.db"
    output:
        "intermediate/{name}.verify.hdf5"
    shell:
        """
        python3 code/plots.py generate-dataset-verify \
            -p simulation-config.toml \
            -b {input} \
            -o {output}
        """


rule verify:
    input:
        "intermediate/{name}.verify.hdf5"
    output:
        static = "intermediate/{name}.verify.static.hdf5",
        logistic = "intermediate/{name}.verify.logistic.hdf5",
    params:
        mpi_nodes = config["mpi_nodes"]
    shell:
        """
        mpirun -n {params.mpi_nodes} ./code/bin/statics-mpi \
            -p simulation-config.toml \
            -i {input} \
            -m static \
            -o {output.static}
        mpirun -n {params.mpi_nodes} ./code/bin/statics-mpi \
            -p simulation-config.toml \
            -i {input} \
            -m logistic \
            -o {output.logistic}
        """


rule abc_diagnostic:
    input:
        up = "data/raw_internal/{upfile}.csv",
        down = "data/raw_internal/{downfile}.csv",
        db = "intermediate/{upfile}.{downfile}.db",
    output:
        "results/figures/{upfile}.{downfile}.abc-diagnostic.pdf"
    shell:
        """
        python3 code/plots.py abcdiag \
            -p simulation-config.toml \
            -b {input.db} \
            -u {input.up} \
            -d {input.down} \
            --save {output}
        """


rule abc_fit:
    input:
        up = "data/raw_internal/{upfile}.csv",
        down = "data/raw_internal/{downfile}.csv",
        db = "intermediate/{upfile}.{downfile}.db",
    output:
        "results/figures/{upfile}.{downfile}.abc-fit.pdf"
    shell:
        """
        python3 code/plots.py abcfit \
            -p simulation-config.toml \
            -b {input.db} \
            -u {input.up} \
            -d {input.down} \
            --save {output}
        """