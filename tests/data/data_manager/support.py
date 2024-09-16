"""Test support for mocking GraphQL queries"""

_list_attrs_resp = {
    "data": {
        "datasetClass": {
            "attributes": ["molecule", "hamiltonian", "sparse_hamiltonian", "hf_state", "full"]
        }
    }
}

_list_datasets_resp = {
    "data": {
        "datasetClasses": {
            "id": "qchem",
            "datasets": [
                {
                    "parameterValues": [
                        {"name": "molname", "value": "H2"},
                        {"name": "bondlength", "value": "1.16"},
                        {"name": "basis", "value": "STO-3G"},
                    ]
                }
            ],
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

_dataclass_ids = {"data": {"datasetClasses": [{"id": "other"}, {"id": "qchem"}, {"id": "qspin"}]}}

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
                {"name": "layout"},
                {"name": "periodicity"},
                {"name": "lattice"},
            ],
            "parameterTree": {
                "next": {
                    "Ising": {
                        "next": {
                            "1x4": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-1x4-chain-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-1x4-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x8": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-1x8-chain-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-1x8-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x2": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-2x2-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-2x2-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-2x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-2x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x8": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-2x8-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-2x8-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "4x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-4x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-4x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x16": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-1x16-chain-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-1x16-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "open": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "Heisenberg": {
                        "next": {
                            "1x4": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x4-chain-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x4-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x8": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x8-chain-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x8-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x2": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x2-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x2-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x8": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x8-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x8-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "4x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-4x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-4x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x16": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x16-chain-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x16-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "open": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "BoseHubbard": {
                        "next": {
                            "1x4": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "bose-hubbard-model-bosehubbard-1x4-chain-open"
                                            },
                                            "closed": {
                                                "value": "bose-hubbard-model-bosehubbard-1x4-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x8": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "bose-hubbard-model-bosehubbard-1x8-chain-open"
                                            },
                                            "closed": {
                                                "value": "bose-hubbard-model-bosehubbard-1x8-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x2": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "bose-hubbard-model-bosehubbard-2x2-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "bose-hubbard-model-bosehubbard-2x2-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "bose-hubbard-model-bosehubbard-2x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "bose-hubbard-model-bosehubbard-2x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "open": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "FermiHubbard": {
                        "next": {
                            "1x4": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "fermi-hubbard-model-fermihubbard-1x4-chain-open"
                                            },
                                            "closed": {
                                                "value": "fermi-hubbard-model-fermihubbard-1x4-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x8": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "fermi-hubbard-model-fermihubbard-1x8-chain-open"
                                            },
                                            "closed": {
                                                "value": "fermi-hubbard-model-fermihubbard-1x8-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x2": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "fermi-hubbard-model-fermihubbard-2x2-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "fermi-hubbard-model-fermihubbard-2x2-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "fermi-hubbard-model-fermihubbard-2x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "fermi-hubbard-model-fermihubbard-2x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "open": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
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

_error_response = {"data": None, "errors": [{"message": "Mock error message."}]}
