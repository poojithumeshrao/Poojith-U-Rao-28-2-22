def circuit():
        """Implements the Deutsch Jozsa algorithm."""

        # QHACK #

        # Insert any pre-oracle processing here

        #qml.PauliX(wires=3)
        #qml.PauliX(wires=4)
        qml.Hadamard(wires = 0)
        qml.Hadamard(wires = 1)

        # for i in [4]:
        #     qml.PauliX(wires = i)
        #     qml.Hadamard(wires = i)
        
        #     fs[i-2]([0,1,i])

        qml.PauliX(wires = 2)
        qml.Hadamard(wires = 2)
        
        qml.Hadamard(wires = 3)
        qml.Hadamard(wires = 4)

        qml.PauliX(wires = 5)
        qml.Hadamard(wires = 5)

        
        qml.ctrl(fs[2],control=[3,4])([0,1,2])
        
        qml.PauliX(wires=3)
        qml.PauliX(wires=4)

        qml.ctrl(fs[0],control=[3,4])([0,1,2])

        qml.PauliX(wires=4)
        
        qml.ctrl(fs[1],control=[3,4])([0,1,2])
        qml.PauliX(wires=3)

        qml.PauliX(wires=4)

        qml.ctrl(fs[3],control=[3,4])([0,1,2])

        qml.PauliX(wires=4)
        
        
        
        # Insert any post-oracle processing here

        qml.Hadamard(wires = 0)
        qml.Hadamard(wires = 1)

        qml.PauliX(wires=0)
        qml.PauliX(wires=1)

        
        qml.Toffoli(wires=[0,1,5])
        qml.Hadamard(wires = 3)
        qml.Hadamard(wires = 4)
        
        # QHACK #

        return (qml.probs(wires=[0,1]))
