import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import DataGrid, { Column } from 'react-data-grid';
import { FaCloudDownloadAlt } from 'react-icons/fa';
import { Link } from 'react-router-dom';
import { MLStatisticsModel } from '../types';
import { DisplayTableStatistics } from './DisplayTableStatistics';

export interface DisplayScoresProps {
  title: string | null;
  scores: MLStatisticsModel;
  modelName?: string;
  projectSlug?: string | null;
}

interface Row {
  id: string;
  label: string;
  prediction: string;
  text: string;
}

/**
 * DisplayScores component to show model statistics and false predictions.
 * It includes a table of statistics and a data grid for false predictions.
 **/
export const DisplayScores: FC<DisplayScoresProps> = ({
  title,
  scores,
  modelName,
  projectSlug,
}) => {
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
  const [showFalsePredictions, setShowFalsePredictions] = useState(false);
  const columns: readonly Column<Row>[] = [
    {
      key: 'id',
      name: 'Id',
      resizable: true,
      width: 180,
      renderCell: (props) => (
        <div>
          {projectSlug ? (
            <Link to={`/projects/${projectSlug}/tag/${props.row.id}`}>{props.row.id}</Link>
          ) : (
            props.row.id
          )}
        </div>
      ),
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
  if (!scores) return;
  return (
    <div>
      <span className="explanations">
        The current model has a f1 macro of <b>{scores.f1_macro}</b>
      </span>
      <DisplayTableStatistics scores={scores} title={title} />
      {scores['false_predictions'] && (
        <button
          className="btn btn-outline-secondary btn-sm me-2 "
          id="false-predictions"
          onClick={() => setShowFalsePredictions(true)}
        >
          Show false predictions
        </button>
      )}
      <button
        className="btn btn-outline-secondary btn-sm me-2"
        id="download-params"
        onClick={(e) => {
          e.preventDefault();
          downloadModel();
        }}
      >
        <FaCloudDownloadAlt size={15} className="me-2" />
        Download as JSON
      </button>
      <Modal
        show={showFalsePredictions}
        id="quickmodel-modal"
        onHide={() => setShowFalsePredictions(false)}
        centered
        size="xl"
      >
        <Modal.Header closeButton>
          <Modal.Title>False prediction of the model {modelName}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {' '}
          <DataGrid<Row>
            className="fill-grid"
            columns={columns}
            rows={scores['false_predictions'] as Row[]}
          />
        </Modal.Body>
      </Modal>
    </div>
  );
};
