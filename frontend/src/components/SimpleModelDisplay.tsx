import { FC } from 'react';
import { DisplayScores } from './DisplayScores';

interface SimpleModelDisplayProps {
  currentModel?: Record<string, never>;
}

export const SimpleModelDisplay: FC<SimpleModelDisplayProps> = ({ currentModel }) => {
  // if no model, return nothing
  if (!currentModel) return null;

  return (
    <div>
      <hr />
      <h5>Current {currentModel.model} model</h5>
      <table className="table table-striped table-hover">
        <tbody>
          {currentModel.params &&
            Object.entries(currentModel.params).map(([key, value], i) => (
              <tr key={i}>
                <td>{key}</td>
                <td>
                  {Array.isArray(value)
                    ? value.join(', ') // or use bullets if you prefer
                    : typeof value === 'object' && value !== null
                      ? JSON.stringify(value, null, 2)
                      : String(value)}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
      <div>
        <h5>Statistics</h5>
        <DisplayScores
          title="Quick model"
          scores={currentModel.statistics as unknown as Record<string, number>}
        />
        {currentModel.statistics_cv10 && (
          <DisplayScores
            title="Cross validation"
            scores={currentModel.statistics_cv10 as unknown as Record<string, number>}
          />
        )}
      </div>
    </div>
  );
};
