import React from 'react';
import {
  VictoryAxis,
  VictoryChart,
  VictoryLegend,
  VictoryLine,
  VictoryScatter,
  VictoryTheme,
} from 'victory';

interface LossChartProps {
  loss: LossData | null;
  xmax?: number;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

export const LossChart: React.FC<LossChartProps> = ({ loss, xmax }) => {
  const val_epochs = loss ? (Object.values(loss.epoch) as unknown as number[]) : [];
  const val_loss = loss ? (Object.values(loss.val_loss) as unknown as number[]) : [];
  const val_eval_loss = loss ? (Object.values(loss.val_eval_loss) as unknown as number[]) : [];

  const valLossData = val_epochs.map((epoch, i) => ({
    x: epoch as number,
    y: val_loss[i] as number,
  }));
  const valEvalLossData = val_epochs.map((epoch, i) => ({
    x: epoch as number,
    y: val_eval_loss[i] as number,
  }));

  if (valEvalLossData.length < 1)
    return (
      <div className="alert alert-info m-3">
        Loss chart will be displayed when enough data is available
      </div>
    );

  return (
    <>
      <VictoryChart
        theme={VictoryTheme.material}
        minDomain={{ y: 0 }}
        maxDomain={{ x: xmax }}
        width={1000}
        height={500}
      >
        <VictoryAxis
          label="Epoch"
          style={{
            axisLabel: { padding: 30 },
          }}
        />
        <VictoryAxis
          dependentAxis
          label="Loss"
          style={{
            axisLabel: { padding: 40 },
          }}
        />
        <VictoryLine
          data={valLossData}
          style={{
            data: { stroke: '#c43a31' }, // Rouge pour val_loss
          }}
        />
        <VictoryScatter
          data={valLossData}
          size={5} // <-- Adjust size here
          style={{
            data: { fill: '#c43a31' },
          }}
        />
        <VictoryLine
          data={valEvalLossData}
          style={{
            data: { stroke: '#0000ff' }, // Bleu pour val_eval_loss
          }}
        />
        <VictoryScatter
          data={valEvalLossData}
          size={5} // <-- Adjust size here
          style={{
            data: { fill: '#0000ff' },
          }}
        />
        <VictoryLegend
          x={100}
          y={350}
          title="Legend"
          centerTitle
          orientation="horizontal"
          gutter={20}
          style={{ border: { stroke: 'black' }, title: { fontSize: 10 } }}
          data={[
            { name: 'Train Loss', symbol: { fill: '#c43a31' } },
            { name: 'Eval Loss', symbol: { fill: '#0000ff' } },
          ]}
          standalone={true}
        />
      </VictoryChart>
    </>
  );
};
