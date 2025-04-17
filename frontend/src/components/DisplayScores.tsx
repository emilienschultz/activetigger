import { FC } from 'react';
import { DisplayMatrix } from './DisplayMatrix';

export interface DisplayScores {
  scores: Record<string, string | number | Record<string, number> | Record<string, string>>;
  scores_cv10?: Record<string, string | number | Record<string, number> | Record<string, string>>;
}

// component
export const DisplayScores: FC<DisplayScores> = ({ scores, scores_cv10 }) => {
  // element to display in the tab
  const scores_filtered = Object.entries(scores).filter(
    ([key]) => key !== 'false_predictions' && key !== 'confusion_matrix',
  );

  // get the labels
  const labels = Object.keys(scores['f1_label'] || []);
  return (
    <table className="table">
      {' '}
      <thead>
        <tr>
          <th scope="col">Key</th>
          <th scope="col">Value</th>
          {scores_cv10 && <th scope="col">Value (CV10)</th>}
        </tr>
      </thead>
      <tbody>
        {Object.values(scores_filtered || {}).map(([key, value], i) => (
          <tr key={i}>
            <td>{key}</td>
            <td>{JSON.stringify(value)}</td>
            {scores_cv10 && <td>{JSON.stringify(scores_cv10?.[key])}</td>}
          </tr>
        ))}
        <tr>
          <td>Confusion matrix</td>
          <td>
            <DisplayMatrix
              matrix={scores['confusion_matrix'] as unknown as number[][]}
              labels={labels}
            />
          </td>
        </tr>
      </tbody>
    </table>
  );
};
