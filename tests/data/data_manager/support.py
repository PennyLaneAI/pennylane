"""Test support for mocking GraphQL queries"""

_list_attrs_resp = {
    "data": {
        "datasetClass": {
            "attributes": [
                {"name": "basis_rot_groupings"},
                {"name": "basis_rot_samples"},
                {"name": "dipole_op"},
                {"name": "fci_energy"},
                {"name": "fci_spectrum"},
                {"name": "hamiltonian"},
                {"name": "hf_state"},
                {"name": "molecule"},
                {"name": "number_op"},
                {"name": "optimal_sector"},
                {"name": "paulix_ops"},
                {"name": "qwc_groupings"},
                {"name": "qwc_samples"},
                {"name": "sparse_hamiltonian"},
                {"name": "spin2_op"},
                {"name": "spinz_op"},
                {"name": "symmetries"},
                {"name": "tapered_dipole_op"},
                {"name": "tapered_hamiltonian"},
                {"name": "tapered_hf_state"},
                {"name": "tapered_num_op"},
                {"name": "tapered_spin2_op"},
                {"name": "tapered_spinz_op"},
                {"name": "vqe_energy"},
                {"name": "vqe_gates"},
                {"name": "vqe_params"},
            ]
        }
    }
}

_list_attrs_mqt_resp = {
    "data": {
        "datasetClass": {
            "attributes": [
                {"name": "ae"},
                {"name": "dj"},
                {"name": "ghz"},
                {"name": "graphstate"},
                {"name": "groundstate"},
                {"name": "grover_noancilla"},
                {"name": "grover_v_chain"},
                {"name": "portfolioqaoa"},
                {"name": "portfoliovqe"},
                {"name": "pricingcall"},
                {"name": "pricingput"},
                {"name": "qaoa"},
                {"name": "qft"},
                {"name": "qftentangled"},
                {"name": "qnn"},
                {"name": "qpeexact"},
                {"name": "qpeinexact"},
                {"name": "qwalk_noancilla"},
                {"name": "qwalk_v_chain"},
                {"name": "random"},
                {"name": "realamprandom"},
                {"name": "routing"},
                {"name": "shor"},
                {"name": "su2random"},
                {"name": "tsp"},
                {"name": "twolocalrandom"},
                {"name": "vqe"},
                {"name": "wstate"},
            ]
        }
    }
}

_list_attrs_max3sat_resp = {
    "data": {
        "datasetClass": {
            "attributes": [
                {"name": "clauses"},
                {"name": "hamiltonians"},
                {"name": "ns"},
                {"name": "ids"},
                {"name": "instanceids"},
                {"name": "ratios"},
            ]
        }
    }
}


_get_urls_resp = {
    "data": {
        "datasetClass": {
            "datasets": [
                {
                    "id": "h2_sto-3g_0.46",
                    "downloadUrl": "https://cloud.pennylane.ai/datasets/download/h2_sto-3g_0.46",
                },
                {
                    "id": "h2_sto-3g_1.16",
                    "downloadUrl": "https://cloud.pennylane.ai/datasets/download/h2_sto-3g_1.16",
                },
                {
                    "id": "h2_sto-3g_1.0",
                    "downloadUrl": "https://cloud.pennylane.ai/datasets/download/h2_sto-3g_1.0",
                },
            ]
        }
    }
}

_rydberggpt_url_resp = {
    "data": {
        "datasetClass": {
            "id": "rydberggpt",
            "datasets": [
                {
                    "id": "rydberggpt",
                    "downloadUrl": "https://cloud.pennylane.ai/datasets/v2/download/rydberggpt",
                }
            ],
        }
    }
}

_dataclass_ids = {"data": {"datasetClasses": [{"id": "other"}, {"id": "qchem"}, {"id": "qspin"}]}}

_error_response = {"data": None, "errors": [{"message": "Mock error message."}]}

_parameter_tree = {
    "data": {
        "datasetClass": {
            "attributes": [
                {"name": "ground_energies"},
                {"name": "ground_states"},
                {"name": "hamiltonians"},
                {"name": "num_phases"},
                {"name": "order_params"},
                {"name": "parameters"},
                {"name": "shadow_basis"},
                {"name": "shadow_meas"},
                {"name": "spin_system"},
            ],
            "parameters": [
                {"name": "sysname"},
                {"name": "periodicity"},
                {"name": "lattice"},
                {"name": "layout"},
            ],
            "parameterTree": {
                "next": {
                    "Ising": {
                        "next": {
                            "open": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "transverse-field-ising-model-ising-open-chain-1x4",
                                            "1x8": "transverse-field-ising-model-ising-open-chain-1x8",
                                            "1x16": "transverse-field-ising-model-ising-open-chain-1x16",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "transverse-field-ising-model-ising-open-rectangular-2x2",
                                            "2x4": "transverse-field-ising-model-ising-open-rectangular-2x4",
                                            "2x8": "transverse-field-ising-model-ising-open-rectangular-2x8",
                                            "4x4": "transverse-field-ising-model-ising-open-rectangular-4x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "transverse-field-ising-model-ising-closed-chain-1x4",
                                            "1x8": "transverse-field-ising-model-ising-closed-chain-1x8",
                                            "1x16": "transverse-field-ising-model-ising-closed-chain-1x16",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "transverse-field-ising-model-ising-closed-rectangular-2x2",
                                            "2x4": "transverse-field-ising-model-ising-closed-rectangular-2x4",
                                            "2x8": "transverse-field-ising-model-ising-closed-rectangular-2x8",
                                            "4x4": "transverse-field-ising-model-ising-closed-rectangular-4x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "Heisenberg": {
                        "next": {
                            "open": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "xxz-heisenberg-model-heisenberg-open-chain-1x4",
                                            "1x8": "xxz-heisenberg-model-heisenberg-open-chain-1x8",
                                            "1x16": "xxz-heisenberg-model-heisenberg-open-chain-1x16",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "xxz-heisenberg-model-heisenberg-open-rectangular-2x2",
                                            "2x4": "xxz-heisenberg-model-heisenberg-open-rectangular-2x4",
                                            "2x8": "xxz-heisenberg-model-heisenberg-open-rectangular-2x8",
                                            "4x4": "xxz-heisenberg-model-heisenberg-open-rectangular-4x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "xxz-heisenberg-model-heisenberg-closed-chain-1x4",
                                            "1x8": "xxz-heisenberg-model-heisenberg-closed-chain-1x8",
                                            "1x16": "xxz-heisenberg-model-heisenberg-closed-chain-1x16",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "xxz-heisenberg-model-heisenberg-closed-rectangular-2x2",
                                            "2x4": "xxz-heisenberg-model-heisenberg-closed-rectangular-2x4",
                                            "2x8": "xxz-heisenberg-model-heisenberg-closed-rectangular-2x8",
                                            "4x4": "xxz-heisenberg-model-heisenberg-closed-rectangular-4x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "BoseHubbard": {
                        "next": {
                            "open": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "bose-hubbard-model-bosehubbard-open-chain-1x4",
                                            "1x8": "bose-hubbard-model-bosehubbard-open-chain-1x8",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "bose-hubbard-model-bosehubbard-open-rectangular-2x2",
                                            "2x4": "bose-hubbard-model-bosehubbard-open-rectangular-2x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "bose-hubbard-model-bosehubbard-closed-chain-1x4",
                                            "1x8": "bose-hubbard-model-bosehubbard-closed-chain-1x8",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "bose-hubbard-model-bosehubbard-closed-rectangular-2x2",
                                            "2x4": "bose-hubbard-model-bosehubbard-closed-rectangular-2x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "FermiHubbard": {
                        "next": {
                            "open": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "fermi-hubbard-model-fermihubbard-open-chain-1x4",
                                            "1x8": "fermi-hubbard-model-fermihubbard-open-chain-1x8",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "fermi-hubbard-model-fermihubbard-open-rectangular-2x2",
                                            "2x4": "fermi-hubbard-model-fermihubbard-open-rectangular-2x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "fermi-hubbard-model-fermihubbard-closed-chain-1x4",
                                            "1x8": "fermi-hubbard-model-fermihubbard-closed-chain-1x8",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "fermi-hubbard-model-fermihubbard-closed-rectangular-2x2",
                                            "2x4": "fermi-hubbard-model-fermihubbard-closed-rectangular-2x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                },
                "default": None,
            },
        }
    }
}


_qchem_parameter_tree = {
    "data": {
        "datasetClass": {
            "attributes": [
                {"name": "basis_rot_groupings"},
                {"name": "basis_rot_samples"},
                {"name": "dipole_op"},
                {"name": "fci_energy"},
                {"name": "fci_spectrum"},
                {"name": "hamiltonian"},
                {"name": "hf_state"},
                {"name": "molecule"},
                {"name": "number_op"},
                {"name": "optimal_sector"},
                {"name": "paulix_ops"},
                {"name": "qwc_groupings"},
                {"name": "qwc_samples"},
                {"name": "sparse_hamiltonian"},
                {"name": "spin2_op"},
                {"name": "spinz_op"},
                {"name": "symmetries"},
                {"name": "tapered_dipole_op"},
                {"name": "tapered_hamiltonian"},
                {"name": "tapered_hf_state"},
                {"name": "tapered_num_op"},
                {"name": "tapered_spin2_op"},
                {"name": "tapered_spinz_op"},
                {"name": "vqe_energy"},
                {"name": "vqe_gates"},
                {"name": "vqe_params"},
            ],
            "parameters": [
                {"name": "molname"},
                {"name": "basis"},
                {"name": "bondlength"},
                {"name": "bondangle"},
                {"name": "number_of_spin_orbitals"},
            ],
            "parameterTree": {
                "next": {
                    "C2": {
                        "next": {
                            "STO-3G": {
                                "next": {
                                    "0.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-0.5-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-0.7-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-0.9-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-1.1-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.3": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-1.3-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-1.5-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-1.7-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-1.9-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-2.1-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.3": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-2.3-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "c2-molecule-c2-sto-3g-2.5-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.246": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "c2-molecule-c2-sto-3g-1.246-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                },
                                "default": "1.246",
                            }
                        },
                        "default": "STO-3G",
                    },
                    "CO": {
                        "next": {
                            "STO-3G": {
                                "next": {
                                    "0.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-0.5-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-0.7-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-0.9-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-1.1-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.3": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-1.3-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-1.5-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-1.7-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-1.9-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-2.1-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.3": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-2.3-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"20": "co-molecule-co-sto-3g-2.5-n\\a-20"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.128": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "co-molecule-co-sto-3g-1.128-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                },
                                "default": "1.128",
                            }
                        },
                        "default": "STO-3G",
                    },
                    "H2": {
                        "next": {
                            "6-31G": {
                                "next": {
                                    "0.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.5-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.7-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.9-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.1-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.3": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.3-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.5-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.7-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.9-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-2.1-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.54": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.54-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.58": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.58-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.62": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.62-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.66": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.66-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.74": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.74-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.78": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.78-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.82": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.82-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.86": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.86-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.94": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.94-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.98": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.98-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.02": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.02-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.06": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.06-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.14": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.14-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.18": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.18-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.22": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.22-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.26": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.26-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.34": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.34-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.38": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.38-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.42": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.42-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.46": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.46-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.54": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.54-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.58": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.58-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.62": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.62-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.66": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.66-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.74": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.74-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.78": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.78-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.82": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.82-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.86": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.86-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.94": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.94-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.98": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-1.98-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.02": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-2.02-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.06": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-2.06-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.742": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"8": "h2-molecule-h2-6-31g-0.742-n\\a-8"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                },
                                "default": "0.742",
                            },
                            "STO-3G": {
                                "next": {
                                    "0.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.5-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.7-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.9-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.1-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.3": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.3-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.5-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.7-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.9-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-2.1-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.54": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.54-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.58": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.58-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.62": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.62-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.66": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.66-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.74": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.74-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.78": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.78-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.82": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.82-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.86": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.86-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.94": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.94-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.98": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.98-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.02": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.02-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.06": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.06-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.14": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.14-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.18": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.18-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.22": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.22-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.26": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.26-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.34": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.34-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.38": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.38-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.42": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.42-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.46": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.46-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.54": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.54-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.58": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.58-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.62": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.62-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.66": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.66-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.74": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.74-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.78": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.78-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.82": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.82-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.86": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.86-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.94": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.94-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.98": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-1.98-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.02": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-2.02-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.06": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-2.06-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.742": {
                                        "next": {
                                            "N\\A": {
                                                "next": {"4": "h2-molecule-h2-sto-3g-0.742-n\\a-4"},
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                },
                                "default": "0.742",
                            },
                            "CC-PVDZ": {
                                "next": {
                                    "0.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-0.5-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-0.7-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-0.9-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-1.1-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.3": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-1.3-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-1.5-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.7": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-1.7-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "1.9": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-1.9-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.1": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-2.1-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.3": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-2.3-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "2.5": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-2.5-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                    "0.742": {
                                        "next": {
                                            "N\\A": {
                                                "next": {
                                                    "20": "h2-molecule-h2-cc-pvdz-0.742-n\\a-20"
                                                },
                                                "default": None,
                                            }
                                        },
                                        "default": None,
                                    },
                                },
                                "default": "0.742",
                            },
                        },
                        "default": "STO-3G",
                    },
                }
            },
        }
    }
}
