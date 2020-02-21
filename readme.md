# Übersicht
Dies ist ein einfacher Reinforcement Learning (RL) Agent, der Bitcoin handeln soll. Diese Repo dient nur dazu, sich in das Thema RL einzuarbeiten (und ein wenig in Python)



# Files
-   `Agent.py`: Stellt den RL Agenten dar
-   `trading_gym.py`: ist die Tradeumgebung, die für das Training erstellt wurde
-   `train.py`: beinhaltet die Funktion, den Agenten zu trainieren
-   `trade.py`: beinhaltet die Funktion, ein Model anzugeben und mit diesem eine Ausgewählte Zeitspanne zu handeln. Dies dient nur zum evaluieren des Modells
-   `trade_multi.py`: ähnlich wie `trade.py`, nur dass viele verschiedene Modelle geleichzeitig (unabhängig) an der fake Börse handeln. 
-   `common.py`: definiert Helfer Funktionen
