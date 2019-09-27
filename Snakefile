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
            -o {output.static} \
            -vvv
        mpirun -n {params.mpi_nodes} ./code/bin/statics-mpi \
            -p simulation-config.toml \
            -i {input} \
            -m logistic \
            -o {output.logistic} \
            -vvv
        """


rule plot_verify_output:
    input:
        infile = "intermediate/{name}.verify.hdf5",
        static = "intermediate/{name}.verify.static.hdf5",
        logistic = "intermediate/{name}.verify.logistic.hdf5",
    output:
        static = "results/figures/{name}.verify.static.pdf",
        logistic = "results/figures/{name}.verify.logistic.pdf",
    shell:
        """
        python3 code/plots.py verification-plots \
            -p simulation-config.toml \
            -i {input.infile} \
            -o {input.static} \
            --save {output.static}
        python3 code/plots.py verification-plots \
            -p simulation-config.toml \
            -i {input.infile} \
            -o {input.logistic} \
            --save {output.logistic}
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


rule make_mpi_input:
    input:
        "intermediate/{name}.db"
    output:
        "intermediate/{name}.mpi-in.hdf5"
    shell:
        """
        python3 code/plots.py generate-dataset-mpi \
            -p simulation-config.toml \
            -b {input} \
            -o {output}
        """


rule mpi:
    input:
        "intermediate/{name}.mpi-in.hdf5"
    output:
        "intermediate/{name}.mpi-out.hdf5"
    params:
        mpi_nodes = config["mpi_nodes"]
    shell:
        """
        mpirun -n {params.mpi_nodes} ./code/bin/dap-mpi \
            -p simulation-config.toml \
            -i {input} \
            -o {output} \
            -vvv
        """


rule plot_mpi_output:
    input:
        infile = "intermediate/{name}.mpi-in.hdf5",
        outfile = "intermediate/{name}.mpi-out.hdf5",
    output:
        "results/figures/{name}.mpi.pdf"
    shell:
        """
        python3 code/plots.py mpiout \
            -p simulation-config.toml \
            -i {input.infile} \
            -o {input.outfile} \
            --save {output}
        """


rule make_holiday_input:
    input:
        "intermediate/{name}.db"
    output:
        "intermediate/{name}.holiday-in.hdf5"
    shell:
        """
        python3 code/plots.py generate-dataset-holiday \
            -p simulation-config.toml \
            -b {input} \
            -o {output}
        """


rule holiday:
    input:
        "intermediate/{name}.holiday-in.hdf5"
    output:
        "intermediate/{name}.holiday-out.hdf5"
    params:
        mpi_nodes = config["mpi_nodes"]
    shell:
        """
        mpirun -n {params.mpi_nodes} ./code/bin/holiday-mpi \
            -p simulation-config.toml \
            -i {input} \
            -o {output} \
            -vvv
        """


rule plot_holiday_output:
    input:
        infile = "intermediate/{name}.holiday-in.hdf5",
        outfile = "intermediate/{name}.holiday-out.hdf5",
    output:
        "results/figures/{name}.holiday.pdf"
    shell:
        """
        python3 code/plots.py holiday-plots \
            -p simulation-config.toml \
            -i {input.infile} \
            -o {input.outfile} \
            --save {output}
        """

