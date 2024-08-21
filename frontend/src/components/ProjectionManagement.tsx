import { pick } from 'lodash';
import { FC, useEffect, useState } from 'react';
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
import { useAppContext } from '../core/context';
import { ProjectionInStrictModel, ProjectionModelParams } from '../types';

// function to generate random colors
// TODO : better selection
const generateRandomColor = () => {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
};

const colormap = [
  '#1f77b4', // tab:blue
  '#ff7f0e', // tab:orange
  '#2ca02c', // tab:green
  '#d62728', // tab:red
  '#9467bd', // tab:purple
  '#8c564b', // tab:brown
  '#e377c2', // tab:pink
  '#7f7f7f', // tab:gray
  '#bcbd22', // tab:olive
  '#17becf', // tab:cyan
];

// define the component
export const ProjectionManagement: FC = () => {
  // hook for all the parameters
  const {
    appContext: { currentProject: project, currentScheme, currentProjection },
    setAppContext,
  } = useAppContext();
  const navigate = useNavigate();
  const { authenticatedUser } = useAuth();

  const projectName = project?.params.project_slug ? project?.params.project_slug : null;

  // fetch projection data with the API (null if no model)
  const { projectionData, reFetchProjectionData } = useGetProjectionData(
    projectName,
    currentScheme,
  );

  // form management
  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableProjections = project?.projections.options ? project?.projections.options : null;

  const { register, handleSubmit, watch } = useForm<ProjectionInStrictModel>({
    defaultValues: {
      method: '',
      features: [],
      params: {
        //common
        n_components: 2,
        // T-SNE
        perplexity: 30,
        learning_rate: 'auto',
        init: 'random',
        // UMAP
        metric: 'euclidean',
        n_neighbors: 15,
        min_dist: 0.1,
      },
    },
  });
  const selectedMethod = watch('method'); // state for the model selected to modify parameters

  // action when form validated
  const { updateProjection } = useUpdateProjection(projectName, currentScheme);
  const onSubmit: SubmitHandler<ProjectionInStrictModel> = async (formData) => {
    // fromData has all fields whatever the selected method

    // discard unrelevant fields depending on selected method
    const relevantParams =
      selectedMethod === 'tsne'
        ? ['perplexity', 'n_components', 'learning_rate', 'init']
        : selectedMethod === 'umap'
          ? ['n_neighbors', 'min_dist', 'metric', 'n_components']
          : [];
    const params = pick(formData.params, relevantParams) as ProjectionModelParams;
    const data = { ...formData, params };
    await updateProjection(data);
  };

  // scatterplot management for colors
  const [labelColorMapping, setLabelColorMapping] = useState<{ [key: string]: string } | null>(
    null,
  );

  // useEffect(() => {
  //   if (projectionData && !labelColorMapping) {
  //     const uniqueLabels = projectionData ? [...new Set(projectionData.labels)] : [];
  //     const colors = uniqueLabels.reduce<{ [key: string]: string }>((acc, label) => {
  //       acc[label as string] = generateRandomColor();
  //       return acc;
  //     }, {});
  //     setLabelColorMapping(colors);
  //   }
  // }, [projectionData, labelColorMapping]);

  useEffect(() => {
    if (projectionData && !labelColorMapping) {
      const uniqueLabels = projectionData ? [...new Set(projectionData.labels)] : [];
      const labeledColors = uniqueLabels.reduce<Record<string, string>>(
        (acc, label, index: number) => {
          acc[label as string] = colormap[index];
          return acc;
        },
        {},
      );
      setLabelColorMapping(labeledColors);
    }
  }, [projectionData, labelColorMapping]);

  // manage projection refresh (could be AMELIORATED)
  useEffect(() => {
    // case a first projection is added
    if (
      project &&
      authenticatedUser &&
      !currentProjection &&
      authenticatedUser?.username in project?.projections.available
    ) {
      reFetchProjectionData();
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData?.status }));
      console.log('Fetch projection data');
    }
    // case if the projection changed
    if (
      authenticatedUser &&
      currentProjection &&
      currentProjection != project?.projections.available[authenticatedUser?.username]
    ) {
      console.log('Refetch projection data');
      reFetchProjectionData();
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData?.status }));
    }
  }, [project, authenticatedUser, currentProjection]);

  // manage zoom selection
  const [zoomDomain, setZoomDomain] = useState(null);
  // console.log(zoomDomain);

  const handleZoom = (domain: any) => {
    setZoomDomain(domain);
  };
  // TODO : add to configuration context

  return (
    <div>
      <details>
        <summary>Configure</summary>
        <form onSubmit={handleSubmit(onSubmit)}>
          <label htmlFor="model">Select a model</label>
          <select id="model" {...register('method')}>
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
          {availableProjections && selectedMethod == 'tsne' && (
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
          {availableProjections && selectedMethod == 'umap' && (
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
            {...register('params.n_components', { valueAsNumber: true, required: true })}
          ></input>

          <button className="btn btn-primary btn-validation">Compute</button>
        </form>
      </details>
      {projectionData && labelColorMapping && (
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
                size={2}
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
                      onClick: (_, props) => {
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
