import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { Link } from 'react-router-dom';
import {
  useGetAnnotationsFile,
  useGetFeaturesFile,
  useGetModelUrl,
  useGetPredictionsFile,
} from '../core/api';
import { useAppContext } from '../core/context';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to display the export page
 */

export const ProjectExportPage: FC = () => {
  const { projectName } = useParams();

  const {
    appContext: { currentProject: project, currentScheme },
  } = useAppContext();

  const [format, setFormat] = useState<string>('csv');
  const [features, setFeatures] = useState<string[] | null>(null);
  const [model, setModel] = useState<string | null>(null);

  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableModels =
    currentScheme && project?.bertmodels.available[currentScheme]
      ? Object.keys(project?.bertmodels.available[currentScheme])
      : [];
  const availablePrediction =
    currentScheme && model
      ? project?.bertmodels.available[currentScheme][model]['predicted']
      : false;

  const { getFeaturesFile } = useGetFeaturesFile(projectName || null);
  const { getAnnotationsFile } = useGetAnnotationsFile(projectName || null);
  const { getPredictionsFile } = useGetPredictionsFile(projectName || null);
  const { modelUrl } = useGetModelUrl(projectName || null, model);

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="export">
      <div className="container-fluid">
        <div className="row">
          <div className="col-8">
            <div className="explanations">
              Download data (annotations, features, predictions) and fine-tuned models
            </div>

            <div>Select a format</div>
            <select
              className="form-select"
              onChange={(e) => {
                setFormat(e.currentTarget.value);
              }}
            >
              <option key="csv">csv</option>
              <option key="xlsx">xlsx</option>
              <option key="parquet">parquet</option>
            </select>
            <h4 className="subsection">Annotations</h4>
            <button
              className="btn btn-primary"
              onClick={() => {
                if (currentScheme) getAnnotationsFile(currentScheme, format, 'train');
              }}
            >
              Export training data
            </button>

            {project?.params.test && (
              <button
                className="btn btn-primary m-3"
                onClick={() => {
                  if (currentScheme) getAnnotationsFile(currentScheme, format, 'test');
                }}
              >
                Export test data
              </button>
            )}

            <h4 className="subsection">Features</h4>
            <div>
              <div>
                <select
                  className="form-select"
                  onChange={(e) => {
                    setFeatures(Array.from(e.target.selectedOptions, (option) => option.value));
                  }}
                  multiple
                >
                  {(availableFeatures || []).map((e) => (
                    <option key={e}>{e}</option>
                  ))}
                </select>
              </div>
              <div>
                <button
                  className="btn btn-primary mt-3"
                  onClick={() => {
                    if (features) {
                      getFeaturesFile(features, format);
                    }
                  }}
                >
                  Export selected features
                </button>
              </div>
              <h4 className="subsection">Fine-tuned models and predictions</h4>
              <div className="explanations">Select first a model</div>
              <div>
                <div>
                  <select
                    className="form-select"
                    onChange={(e) => {
                      setModel(e.target.value);
                    }}
                  >
                    <option></option>
                    {(availableModels || []).map((e) => (
                      <option key={e}>{e}</option>
                    ))}
                  </select>
                </div>
                <div>
                  {modelUrl && (
                    <Link to={modelUrl} target="_blank" download className="btn btn-primary mt-3">
                      Export fine-tuned model
                    </Link>
                  )}
                  {/* <button
                    className="btn btn-primary mt-3"
                    onClick={() => {
                      if (model) {
                        getModelFile(model);
                      }
                    }}
                  >
                    Export model
                  </button> */}
                </div>
                <div>
                  {availablePrediction && (
                    <button
                      className="btn btn-primary mt-3"
                      onClick={() => {
                        if (model) {
                          getPredictionsFile(model, format);
                        }
                      }}
                    >
                      Export dataset prediction
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
