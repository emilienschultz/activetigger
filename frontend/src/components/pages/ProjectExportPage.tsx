import { FC, useState } from 'react';
import { Link, useParams } from 'react-router-dom';

import {
  useGetAnnotationsFile,
  useGetFeaturesFile,
  useGetModelFile,
  useGetPredictionsFile,
  useGetPredictionsSimplemodelFile,
  useGetProjectionFile,
  useGetRawDataFile,
  useGetStaticUrls,
} from '../../core/api';
import { useAuth } from '../../core/auth';
import config from '../../core/config';
import { useAppContext } from '../../core/context';
import { ProjectPageLayout } from '../layout/ProjectPageLayout';

/**
 * Component to display the export page
 */

export const ProjectExportPage: FC = () => {
  const { projectName } = useParams();

  // get the current state of the project
  const {
    appContext: { currentProject: project, currentScheme },
  } = useAppContext();
  const { authenticatedUser } = useAuth();

  const [format, setFormat] = useState<string>('csv');
  const [features, setFeatures] = useState<string[] | null>(null);
  const [model, setModel] = useState<string | null>(null);

  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableProjection =
    authenticatedUser?.username && project?.projections.available[authenticatedUser?.username]
      ? project?.projections.available[authenticatedUser?.username]
      : null;
  const availableModels =
    currentScheme && project?.languagemodels.available[currentScheme]
      ? Object.keys(project?.languagemodels.available[currentScheme])
      : [];
  const availablePrediction =
    (currentScheme &&
      model &&
      project?.languagemodels?.available?.[currentScheme]?.[model]?.['predicted']) ??
    false;
  const availablePredictionExternal =
    (currentScheme &&
      model &&
      project?.languagemodels?.available?.[currentScheme]?.[model]?.['predicted_external']) ??
    false;

  const { getFeaturesFile } = useGetFeaturesFile(projectName || null);
  const { getAnnotationsFile } = useGetAnnotationsFile(projectName || null);
  const { getPredictionsFile } = useGetPredictionsFile(projectName || null);
  const { getModelFile } = useGetModelFile(projectName || null);
  const { getRawDataFile } = useGetRawDataFile(projectName || null);
  const { getPredictionsSimpleModelFile } = useGetPredictionsSimplemodelFile(projectName || null);
  const { getProjectionFile } = useGetProjectionFile(projectName || null);
  const { staticUrls } = useGetStaticUrls(projectName || null, model);

  const isSimpleModel =
    authenticatedUser &&
    currentScheme &&
    project?.simplemodel.available[authenticatedUser.username]?.[currentScheme];

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
              Export train tags
            </button>

            {project?.params.test && (
              <button
                className="btn btn-primary m-3"
                onClick={() => {
                  if (currentScheme) getAnnotationsFile(currentScheme, format, 'test');
                }}
              >
                Export test tags
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

              {availableProjection && (
                <div>
                  <button
                    className="btn btn-primary mt-3"
                    onClick={() => {
                      if (availableProjection) {
                        getProjectionFile(format);
                      }
                    }}
                  >
                    Export current projection
                  </button>
                </div>
              )}

              <h4 className="subsection">Fine-tuned models and predictions</h4>

              {isSimpleModel && (
                <button
                  className="btn btn-primary"
                  onClick={() => {
                    if (currentScheme) getPredictionsSimpleModelFile(currentScheme, format);
                  }}
                >
                  Export simplemodel predictions
                </button>
              )}

              <div className="explanations">For BERT, select first a model</div>
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
                  {/*
            small fix for the direct link when no nging
            */}
                  {model &&
                    (staticUrls ? (
                      <Link
                        to={config.api.url.replace(/\/$/, '') + '/static/' + staticUrls.model?.path}
                        target="_blank"
                        download
                        className="btn btn-primary mt-3"
                      >
                        Export fine-tuned model (static)
                      </Link>
                    ) : (
                      <button
                        className="btn btn-primary mt-3"
                        onClick={() => {
                          getModelFile(model);
                        }}
                      >
                        Export fine-tuned model
                      </button>
                    ))}
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
                      Export prediction complete dataset
                    </button>
                  )}
                </div>
                <div>
                  {availablePredictionExternal && (
                    <button
                      className="btn btn-primary mt-3"
                      onClick={() => {
                        if (model) {
                          getPredictionsFile(model, format, 'external');
                        }
                      }}
                    >
                      Export prediction external dataset
                    </button>
                  )}
                </div>
              </div>
            </div>
            <hr />

            {/*
            small fix for the direct link when no nging
            */}
            {staticUrls ? (
              <Link
                to={config.api.url.replace(/\/$/, '') + '/static/' + staticUrls.dataset.path}
                target="_blank"
                download
                className="btn btn-primary mt-3"
              >
                Export raw dataset in parquet (static)
              </Link>
            ) : (
              <button
                className="btn btn-primary mt-3"
                onClick={() => {
                  getRawDataFile();
                }}
              >
                Export raw dataset in parquet
              </button>
            )}
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
