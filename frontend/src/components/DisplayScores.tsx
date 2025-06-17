import { FC } from 'react';
import DataGrid, { Column } from 'react-data-grid';
import { MLStatisticsModel } from '../types';
import { DisplayTableStatistics } from './DisplayTableStatistics';

export interface DisplayScoresProps {
  title: string | null;
  scores: MLStatisticsModel;
  modelName?: string;
}

interface Row {
  id: string;
  label: string;
  prediction: string;
  text: string;
}

const columns: readonly Column<Row>[] = [
  {
    name: 'Id',
    key: 'id',
    resizable: true,
  },
  {
    name: 'Label',
    key: 'label',
    resizable: true,
  },
  {
    name: 'Prediction',
    key: 'prediction',
    resizable: true,
  },
  {
    name: 'Text',
    key: 'text',
    resizable: true,
  },
];

/**
 * DisplayScores component to show model statistics and false predictions.
 * It includes a table of statistics and a data grid for false predictions.
 **/
export const DisplayScores: FC<DisplayScoresProps> = ({ title, scores, modelName }) => {
  const downloadModel = () => {
    if (!scores) return; // Ensure model is not null or undefined

    // Convert the model object to a JSON string
    const modelJson = JSON.stringify(scores, null, 2);

    // Create a Blob from the JSON string
    const blob = new Blob([modelJson], { type: 'application/json' });

    // Create a temporary link element
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = modelName || 'model.json';
    link.click();
  };
  if (!scores) return;
  return (
    <div>
      <span className="explanations">
        The current model has a f1 macro of <b>{scores.f1_macro}</b>
      </span>
      <DisplayTableStatistics scores={scores} title={title} />
      {scores['false_predictions'] && (
        <details>
          <summary>False predictions</summary>
          <DataGrid<Row>
            className="fill-grid"
            columns={columns}
            rows={scores['false_predictions'] as Row[]}
          />
        </details>
      )}
      <a
        href="#"
        onClick={(e) => {
          e.preventDefault();
          downloadModel();
        }}
      >
        JSON file
      </a>
    </div>
  );
};
