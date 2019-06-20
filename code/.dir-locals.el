;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((c++-mode
  . ((flycheck-gcc-language-standard . "c++17")
     (helm-make-build-dir . "bin")
     (eval . (setq flycheck-gcc-include-path
                    (list (substitute-in-file-name "$CONDA_PREFIX/envs/dual-adaptation/include/"))))
     )))
