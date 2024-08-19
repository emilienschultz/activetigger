import { FC, useEffect, useMemo, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import {
  VictoryChart,
  VictoryLegend,
  VictoryScatter,
  VictoryTheme,
  VictoryTooltip,
  VictoryZoomContainer,
} from 'victory';

import { useGetProjectionData, useUpdateProjection } from '../core/api';
import { useAuth } from '../core/auth';
import { ProjectStateModel, ProjectionInModel } from '../types';

interface ProjectionManagementProps {
  projectName: string;
  currentScheme: string;
  project: ProjectStateModel;
}

// function to generate random colors
const generateRandomColor = () => {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
};

// define the component
export const ProjectionManagement: FC<ProjectionManagementProps> = ({
  projectName,
  currentScheme,
  project,
}) => {
  const { authenticatedUser } = useAuth();
  if (!authenticatedUser?.username) return null;

  const navigate = useNavigate();

  // get projection data (null if no model)
  const { projectionData, reFetchProjectionData } = useGetProjectionData(
    projectName,
    currentScheme,
  );

  console.log(projectionData);

  // form management
  // state for the model selected
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableProjections = project?.projections.options ? project?.projections.options : null;
  const { register, handleSubmit } = useForm<ProjectionInModel>({
    defaultValues: {
      method: '',
      features: [],
      params: {
        n_components: 2,
        perplexity: 30,
        learning_rate: 'auto',
        init: 'random',
        metric: 'euclidean',
        n_neighbors: 15,
        min_dist: 0.1,
      },
    },
  });

  // action when form validated

  const { updateProjection } = useUpdateProjection(projectName, currentScheme);

  const onSubmit: SubmitHandler<ProjectionInModel> = async (formData) => {
    await updateProjection(formData);
  };

  const uniqueLabels = projectionData ? [...new Set(projectionData.labels)] : null;

  const labelColorMapping = useMemo(() => {
    return uniqueLabels
      ? uniqueLabels.reduce<{ [key: string]: string }>((acc, label) => {
          acc[label as string] = generateRandomColor();
          return acc;
        }, {})
      : {};
  }, [reFetchProjectionData]); // Le calcul ne sera refait que si uniqueLabels change

  // manage zoom selection
  const [zoomDomain, setZoomDomain] = useState(null);
  const handleZoom = (domain: any) => {
    setZoomDomain(domain);
  };

  console.log(zoomDomain);

  return (
    <div>
      <div>
        <form onSubmit={handleSubmit(onSubmit)}>
          <label htmlFor="model">Select a model</label>
          <select
            id="model"
            {...register('method')}
            onChange={(e) => {
              setSelectedModel(e.currentTarget.value);
            }}
          >
            <option value=""></option>
            {Object.keys(availableProjections ? availableProjections : []).map((e) => (
              <option key={e} value={e}>
                {e}
              </option>
            ))}{' '}
          </select>
          <div>
            <label htmlFor="features">Select features</label>
            <select id="features" {...register('features')} multiple>
              {Object.values(availableFeatures).map((e) => (
                <option key={e} value={e}>
                  {e}
                </option>
              ))}{' '}
            </select>
          </div>
          {availableProjections && selectedModel == 'tsne' && (
            <div>
              <label htmlFor="perplexity">perplexity</label>
              <input
                type="number"
                step="1"
                id="perplexity"
                {...register('params.perplexity', { valueAsNumber: true })}
              ></input>
              <label>Learning rate</label>
              <select {...register('params.learning_rate')}>
                <option key="auto" value="auto">
                  auto
                </option>
              </select>
              <label>Init</label>
              <select {...register('params.init')}>
                <option key="random" value="random">
                  random
                </option>
              </select>
            </div>
          )}
          {availableProjections && selectedModel == 'umap' && (
            <div>
              <label htmlFor="n_neighbors">n_neighbors</label>
              <input
                type="number"
                step="1"
                id="n_neighbors"
                {...register('params.n_neighbors', { valueAsNumber: true })}
              ></input>
              <label htmlFor="min_dist">min_dist</label>
              <input
                type="number"
                id="min_dist"
                step="0.01"
                {...register('params.min_dist', { valueAsNumber: true })}
              ></input>
              <label htmlFor="metric">Metric</label>
              <select {...register('params.metric')}>
                <option key="euclidean" value="euclidean">
                  euclidean
                </option>
              </select>
            </div>
          )}
          <label htmlFor="n_components">n_components</label>
          <input
            type="number"
            id="n_components"
            step="1"
            {...register('params.n_components', { valueAsNumber: true })}
          ></input>

          <button className="btn btn-primary btn-validation">Compute</button>
        </form>
      </div>
      {projectionData && (
        <div style={{ height: 500, padding: 30 }}>
          {
            <VictoryChart
              theme={VictoryTheme.material}
              containerComponent={<VictoryZoomContainer onZoomDomainChange={handleZoom} />}
              height={300}
              width={300}
            >
              <VictoryScatter
                style={{
                  data: {
                    fill: ({ datum }) => labelColorMapping[datum.labels],
                    opacity: 0.7,
                  },
                }}
                size={1}
                labels={({ datum }) => datum.index}
                labelComponent={
                  <VictoryTooltip style={{ fontSize: 10 }} flyoutStyle={{ fill: 'white' }} />
                }
                data={projectionData.x.map((value, index) => {
                  return {
                    x: value,
                    y: projectionData.y[index],
                    labels: projectionData.labels[index],
                    texts: projectionData.texts[index],
                    index: projectionData.index[index],
                  };
                })}
                events={[
                  {
                    target: 'data',
                    eventHandlers: {
                      onClick: (event, props) => {
                        const { datum } = props;
                        navigate(`/projects/test3/annotate/${datum.index}`);
                      },
                    },
                  },
                ]}
              />
              <VictoryLegend
                x={125}
                y={0}
                title="Legend"
                centerTitle
                orientation="horizontal"
                gutter={10}
                style={{
                  border: { stroke: 'black' },
                  title: { fontSize: 5 },
                  labels: { fontSize: 5 },
                }}
                data={Object.keys(labelColorMapping).map((label) => ({
                  name: label,
                  symbol: { fill: labelColorMapping[label] },
                }))}
              />
            </VictoryChart>
          }
        </div>
      )}
    </div>
  );
};
